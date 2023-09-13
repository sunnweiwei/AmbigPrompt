import redis
import pickle


class TextPassage:
    def __init__(self, collection='wiki-text', host='localhost'):
        self.db = redis.Redis(host=host)
        self.pipe = self.db.pipeline()
        self.collection = collection

    def write(self, documents):
        step = 1000000
        for j, line in enumerate(documents):
            key = self.collection + str(line['pid'])
            value = pickle.dumps(line)
            self.pipe.set(key, value)
            if j % step == 0 and j != 0:
                print(j / len(documents), 'execute')
                self.pipe.execute()
        self.pipe.execute()

    def get_by_id(self, item):
        return self[item - 1]

    def __getitem__(self, item):
        return pickle.loads(self.db.get(self.collection + str(item)))

    def __len__(self):
        return 21015324 if self.collection == 'wiki-text' else 24276193


import types
import torch
import transformers
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
import torch.nn.functional as F
from torch import nn


class FiDT5(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.wrap_encoder()

    def forward_(self, **kwargs):
        if 'input_ids' in kwargs:
            kwargs['input_ids'] = kwargs['input_ids'].view(kwargs['input_ids'].size(0), -1)
        if 'attention_mask' in kwargs:
            kwargs['attention_mask'] = kwargs['attention_mask'].view(kwargs['attention_mask'].size(0), -1)

        return super(FiDT5, self).forward(
            **kwargs
        )

    # We need to resize as B x (N * L) instead of (B * N) x L here
    # because the T5 forward method uses the input tensors to infer
    # dimensions used in the decoder.
    # EncoderWrapper resizes the inputs as (B * N) x L.
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if input_ids != None:
            # inputs might have already be resized in the generate method
            if input_ids.dim() == 3:
                self.encoder.n_passages = input_ids.size(1)
            input_ids = input_ids.view(input_ids.size(0), -1)
        if attention_mask != None:
            attention_mask = attention_mask.view(attention_mask.size(0), -1)
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

    # We need to resize the inputs here, as the generate method expect 2D tensors
    def generate(self, input_ids, attention_mask, max_length):
        self.encoder.n_passages = input_ids.size(1)
        return super().generate(
            input_ids=input_ids.view(input_ids.size(0), -1),
            attention_mask=attention_mask.view(attention_mask.size(0), -1),
            max_length=max_length
        )

    def wrap_encoder(self, use_checkpoint=False):
        """
        Wrap T5 encoder to obtain a Fusion-in-Decoder model.
        """
        self.encoder = EncoderWrapper(self.encoder, use_checkpoint=use_checkpoint)

    def unwrap_encoder(self):
        """
        Unwrap Fusion-in-Decoder encoder, useful to load T5 weights.
        """
        self.encoder = self.encoder.encoder
        block = []
        for mod in self.encoder.block:
            block.append(mod.module)
        block = nn.ModuleList(block)
        self.encoder.block = block

    def load_t5(self, state_dict):
        self.unwrap_encoder()
        self.load_state_dict(state_dict)
        self.wrap_encoder()

    def set_checkpoint(self, use_checkpoint):
        """
        Enable or disable checkpointing in the encoder.
        See https://pytorch.org/docs/stable/checkpoint.html
        """
        for mod in self.encoder.encoder.block:
            mod.use_checkpoint = use_checkpoint

    def reset_score_storage(self):
        """
        Reset score storage, only used when cross-attention scores are saved
        to train a retriever.
        """
        for mod in self.decoder.block:
            mod.layer[1].EncDecAttention.score_storage = None

    def get_crossattention_scores(self, context_mask):
        """
        Cross-attention scores are aggregated to obtain a single scalar per
        passage. This scalar can be seen as a similarity score between the
        question and the input passage. It is obtained by averaging the
        cross-attention scores obtained on the first decoded token over heads,
        layers, and tokens of the input passage.

        More details in Distilling Knowledge from Reader to Retriever:
        https://arxiv.org/abs/2012.04584.
        """
        scores = []
        n_passages = context_mask.size(1)
        for mod in self.decoder.block:
            scores.append(mod.layer[1].EncDecAttention.score_storage)
        scores = torch.cat(scores, dim=2)
        bsz, n_heads, n_layers, _ = scores.size()
        # batch_size, n_head, n_layers, n_passages, text_maxlength
        scores = scores.view(bsz, n_heads, n_layers, n_passages, -1)
        scores = scores.masked_fill(~context_mask[:, None, None], 0.)
        scores = scores.sum(dim=[1, 2, 4])
        ntokens = context_mask.sum(dim=[2]) * n_layers * n_heads
        scores = scores / ntokens
        return scores

    def overwrite_forward_crossattention(self):
        """
        Replace cross-attention forward function, only used to save
        cross-attention scores.
        """
        for mod in self.decoder.block:
            attn = mod.layer[1].EncDecAttention
            attn.forward = types.MethodType(cross_attention_forward, attn)


class EncoderWrapper(torch.nn.Module):
    """
    Encoder Wrapper for T5 Wrapper to obtain a Fusion-in-Decoder model.
    """

    def __init__(self, encoder, use_checkpoint=False):
        super().__init__()

        self.encoder = encoder
        apply_checkpoint_wrapper(self.encoder, use_checkpoint)

    def forward(self, input_ids=None, attention_mask=None, **kwargs, ):
        # total_length = n_passages * passage_length
        bsz, total_length = input_ids.shape
        passage_length = total_length // self.n_passages
        input_ids = input_ids.view(bsz * self.n_passages, passage_length)
        attention_mask = attention_mask.view(bsz * self.n_passages, passage_length)
        kwargs['return_dict'] = True
        outputs = self.encoder(input_ids, attention_mask, **kwargs)
        # outputs = (outputs[0].view(bsz, self.n_passages*passage_length, -1), ) + outputs[1:]
        outputs.last_hidden_state = outputs.last_hidden_state.view(bsz, self.n_passages * passage_length, -1)
        return outputs


class CheckpointWrapper(torch.nn.Module):
    """
    Wrapper replacing None outputs by empty tensors, which allows the use of
    checkpointing.
    """

    def __init__(self, module, use_checkpoint=False):
        super().__init__()
        self.module = module
        self.use_checkpoint = use_checkpoint

    def forward(self, hidden_states, attention_mask, position_bias, **kwargs):
        if self.use_checkpoint and self.training:
            kwargs = {k: v for k, v in kwargs.items() if v is not None}

            def custom_forward(*inputs):
                output = self.module(*inputs, **kwargs)
                empty = torch.tensor(
                    [],
                    dtype=torch.float,
                    device=output[0].device,
                    requires_grad=True)
                output = tuple(x if x is not None else empty for x in output)
                return output

            output = torch.utils.checkpoint.checkpoint(
                custom_forward,
                hidden_states,
                attention_mask,
                position_bias
            )
            output = tuple(x if x.size() != 0 else None for x in output)
        else:
            output = self.module(hidden_states, attention_mask, position_bias, **kwargs)
        return output


def apply_checkpoint_wrapper(t5stack, use_checkpoint):
    """
    Wrap each block of the encoder to enable checkpointing.
    """
    block = []
    for mod in t5stack.block:
        wrapped_mod = CheckpointWrapper(mod, use_checkpoint)
        block.append(wrapped_mod)
    block = nn.ModuleList(block)
    t5stack.block = block


def cross_attention_forward(
        self,
        input,
        mask=None,
        kv=None,
        position_bias=None,
        past_key_value_state=None,
        head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
):
    """
    This only works for computing cross attention over the input
    """
    assert (kv != None)
    assert (head_mask == None)
    assert (position_bias != None or self.has_relative_attention_bias)

    bsz, qlen, dim = input.size()
    n_heads, d_heads = self.n_heads, self.d_kv
    klen = kv.size(1)

    q = self.q(input).view(bsz, -1, n_heads, d_heads).transpose(1, 2)
    if past_key_value_state == None:
        k = self.k(kv).view(bsz, -1, n_heads, d_heads).transpose(1, 2)
        v = self.v(kv).view(bsz, -1, n_heads, d_heads).transpose(1, 2)
    else:
        k, v = past_key_value_state

    scores = torch.einsum("bnqd,bnkd->bnqk", q, k)

    if mask is not None:
        scores += mask

    if position_bias is None:
        position_bias = self.compute_bias(qlen, klen)
    scores += position_bias

    if self.score_storage is None:
        self.score_storage = scores

    attn = F.softmax(scores.float(), dim=-1).type_as(scores)
    attn = F.dropout(attn, p=self.dropout, training=self.training)

    output = torch.matmul(attn, v)
    output = output.transpose(1, 2).contiguous().view(bsz, -1, self.inner_dim)
    output = self.o(output)

    if use_cache:
        output = (output,) + ((k, v),)
    else:
        output = (output,) + (None,)

    if output_attentions:
        output = output + (attn,)

    if self.has_relative_attention_bias:
        output = output + (position_bias,)

    return output


class RetrieverConfig(transformers.BertConfig):

    def __init__(self,
                 indexing_dimension=768,
                 apply_question_mask=False,
                 apply_passage_mask=False,
                 extract_cls=False,
                 passage_maxlength=200,
                 question_maxlength=40,
                 projection=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.indexing_dimension = indexing_dimension
        self.apply_question_mask = apply_question_mask
        self.apply_passage_mask = apply_passage_mask
        self.extract_cls = extract_cls
        self.passage_maxlength = passage_maxlength
        self.question_maxlength = question_maxlength
        self.projection = projection


class Retriever(transformers.PreTrainedModel):
    config_class = RetrieverConfig
    base_model_prefix = "retriever"

    def __init__(self, config, initialize_wBERT=False):
        super().__init__(config)
        assert config.projection or config.indexing_dimension == 768, \
            'If no projection then indexing dimension must be equal to 768'
        self.config = config
        if initialize_wBERT:
            self.model = transformers.BertModel.from_pretrained('bert-base-uncased')
        else:
            self.model = transformers.BertModel(config)
        if self.config.projection:
            self.proj = nn.Linear(
                self.model.config.hidden_size,
                self.config.indexing_dimension
            )
            self.norm = nn.LayerNorm(self.config.indexing_dimension)
        self.loss_fct = torch.nn.KLDivLoss()

    def forward(self,
                question_ids,
                question_mask,
                passage_ids,
                passage_mask,
                gold_score=None):
        question_output = self.embed_text(
            text_ids=question_ids,
            text_mask=question_mask,
            apply_mask=self.config.apply_question_mask,
            extract_cls=self.config.extract_cls,
        )
        bsz, n_passages, plen = passage_ids.size()
        passage_ids = passage_ids.view(bsz * n_passages, plen)
        passage_mask = passage_mask.view(bsz * n_passages, plen)
        passage_output = self.embed_text(
            text_ids=passage_ids,
            text_mask=passage_mask,
            apply_mask=self.config.apply_passage_mask,
            extract_cls=self.config.extract_cls,
        )

        score = torch.einsum(
            'bd,bid->bi',
            question_output,
            passage_output.view(bsz, n_passages, -1)
        )
        score = score / np.sqrt(question_output.size(-1))
        if gold_score is not None:
            loss = self.kldivloss(score, gold_score)
        else:
            loss = None

        return question_output, passage_output, score, loss

    def embed_text(self, text_ids, text_mask, apply_mask=False, extract_cls=False):
        text_output = self.model(
            input_ids=text_ids,
            attention_mask=text_mask if apply_mask else None
        )
        if type(text_output) is not tuple:
            text_output.to_tuple()
        text_output = text_output[0]
        if self.config.projection:
            text_output = self.proj(text_output)
            text_output = self.norm(text_output)

        if extract_cls:
            text_output = text_output[:, 0]
        else:
            if apply_mask:
                text_output = text_output.masked_fill(~text_mask[:, :, None], 0.)
                text_output = torch.sum(text_output, dim=1) / torch.sum(text_mask, dim=1)[:, None]
            else:
                text_output = torch.mean(text_output, dim=1)
        return text_output

    def kldivloss(self, score, gold_score):
        gold_score = torch.softmax(gold_score, dim=-1)
        score = torch.nn.functional.log_softmax(score, dim=-1)
        return self.loss_fct(score, gold_score)


import os
import re
import json
import string
import argparse
import numpy as np
from collections import Counter, defaultdict

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu

tokenizer = PTBTokenizer()


class QAPairEvaluation(object):

    def __init__(self, reference, prediction, metrics="all"):
        '''
        :param: samples: a list of annotated data
        :param: predictions: a dictionary with id as key and prediction as value
                        prediction can be either
                        - a list of strings
                        - a list of dictionaries with quetion and answer as keys
        '''
        self.reference = reference
        self.prediction = [prediction[sample["id"]] for sample in reference]
        self.metrics = metrics
        METRICS_ANSWER = ["F1 answer"]
        METRICS_QG = ["F1 bleu1", "F1 bleu2", "F1 bleu3", "F1 bleu4", "F1 edit-f1"]

        if metrics == "all" and type(self.prediction[0][0]) == str:
            self.metrics = METRICS_ANSWER
        elif metrics == "all":
            self.metrics = METRICS_ANSWER + METRICS_QG

        assert len(set(self.metrics) - set(METRICS_ANSWER) - set(METRICS_QG)) == 0
        self.QG_METRICS_TO_COMPUTE = [m for m in ["bleu1", "bleu2", "bleu3", "bleu4", "rouge-l", "edit-f1"] if
                                      any([metric.endswith(m) for metric in self.metrics])]

        if len(self.QG_METRICS_TO_COMPUTE) > 0:
            # if evaluating QG, tokenize prompt question,
            # reference question and predicted question
            data_to_tokenize = {}
            for i, ref in enumerate(self.reference):
                data_to_tokenize["prompt.{}".format(i)] = [{"caption": ref["question"]}]
                for j, annotation in enumerate(ref["annotations"]):
                    if annotation['type'] == 'multipleQAs':
                        for k, pair in enumerate(annotation['qaPairs']):
                            data_to_tokenize["ref.{}.{}.{}".format(i, j, k)] = \
                                [{'caption': sent.strip()} for sent in pair["question"].split('|') if
                                 len(sent.strip()) > 0]
            for i, pred in enumerate(self.prediction):
                for j, pair in enumerate(pred):
                    data_to_tokenize["gen.{}.{}".format(i, j)] = [{"caption": pair["question"]}]

            all_tokens = tokenizer.tokenize(data_to_tokenize)
            for key, values in all_tokens.items():
                values = {'sent': [normalize_answer(value) for value in values]}
                if key.startswith("prompt."):
                    i = key.split(".")[1]
                    self.reference[int(i)]["question"] = values
                elif key.startswith("ref."):
                    i, j, k = key.split('.')[1:]
                    self.reference[int(i)]["annotations"][int(j)]["qaPairs"][int(k)]["question"] = values
                elif key.startswith("gen."):
                    i, j = key.split(".")[1:]
                    self.prediction[int(i)][int(j)]["question"] = values
                else:
                    raise NotImplementedError()

        self.is_multi = [not any([ann["type"] == "singleAnswer" for ann in ref["annotations"]]) \
                         for ref in self.reference]
        self.results = [self.get_all_metrics(idx) for idx in range(len(self.reference))]

    def print_all_metrics(self):
        for metric in self.metrics:
            result = [e[metric] for e in self.results]
            result_multi_only = [e[metric] for e, is_multi in zip(self.results, self.is_multi) \
                                 if is_multi]
            if metric == "F1 answer":
                print("%s\t%.3f (all)\t%.3f (multi only)" % (metric, np.mean(result), np.mean(result_multi_only)))
            else:
                print("%s\t%.3f" % (metric, np.mean(result_multi_only)))

    def get_metric(self, metric):
        return np.mean([e[metric] for e in self.results])

    def get_all_metrics(self, idx):
        evaluation = {}
        promptQuestion = self.reference[idx]["question"]
        annotations = self.reference[idx]["annotations"]
        if type(self.prediction[idx][0]) == dict:
            # prediction contains a set of question-answer pairs
            predictions = [pair["answer"] for pair in self.prediction[idx]]
            questions = [pair["question"] for pair in self.prediction[idx]]
        else:
            # prediction contains a set of answers
            predictions = self.prediction[idx]
            questions = None

        for annotation in annotations:
            # iterate each annotation and take the maximum metrics
            if annotation['type'] == 'singleAnswer':
                f1 = get_f1([annotation['answer']], predictions)
                for metric in self.metrics:
                    if metric.startswith('F1'):
                        evaluation[metric] = max(evaluation.get(metric, 0), f1)
            elif annotation['type'] == 'multipleQAs':
                matching_pairs = []
                evaluation['F1 answer'] = max(evaluation.get("F1 answer", 0),
                                              get_f1([answer['answer'] for answer in annotation['qaPairs']],
                                                     predictions))
                if questions is None:
                    # skip the below if not evaluating QG
                    continue
                for i, answer in enumerate(annotation["qaPairs"]):
                    for j, prediction in enumerate(predictions):
                        # get every reference-prediction pair with the correct answer prediction
                        em = get_exact_match(answer['answer'], prediction)
                        if em:
                            qg_evals = get_qg_metrics(questions[j],
                                                      answer['question'],
                                                      promptQuestion,
                                                      self.QG_METRICS_TO_COMPUTE)
                            matching_pairs.append((i, j, qg_evals))

                def _get_qg_f1(metric_func):
                    curr_matching_pairs = sorted(matching_pairs, key=lambda x: metric_func(x[2]), reverse=True)
                    occupied_answers = [False for _ in annotation["qaPairs"]]
                    occupied_predictions = [False for _ in predictions]
                    tot = 0
                    # find non-overapping reference-prediction pairs
                    # that match the answer prediction
                    # to get the evaluation score
                    for (i, j, e) in curr_matching_pairs:
                        if occupied_answers[i] or occupied_predictions[j]:
                            continue
                        occupied_answers[i] = True
                        occupied_predictions[j] = True
                        tot += metric_func(e)
                    assert np.sum(occupied_answers) == np.sum(occupied_predictions)
                    return 2 * tot / (len(occupied_answers) + len(occupied_predictions))

                for metric in self.QG_METRICS_TO_COMPUTE:
                    metric_name = "F1 {}".format(metric)
                    if metric_name in self.metrics:
                        e = _get_qg_f1(lambda x: x[metric])
                        evaluation[metric_name] = max(evaluation.get(metric_name, 0), e)
            else:
                raise NotImplementedError()

        assert len(self.metrics) == len(evaluation), (self.metrics, evaluation.keys())
        return evaluation


def get_qg_metrics(generated, question, promptQuestion, metrics):
    evaluation = {}

    # computing bleu scores
    for name, score in zip(['bleu{}'.format(i) for i in range(1, 5)],
                           Bleu(4).compute_score(question, generated)[0]):
        if name in metrics:
            evaluation[name] = score

    # computing edit-f1 score
    if 'edit-f1' in metrics:
        def _get_edits(tokens1, tokens2):
            allCommon = []
            while True:
                commons = list(set(tokens1) & set(tokens2))
                if len(commons) == 0:
                    break
                allCommon += commons
                for c in commons:
                    ind1, ind2 = tokens1.index(c), tokens2.index(c)
                    tokens1 = tokens1[:ind1] + tokens1[ind1 + 1:]
                    tokens2 = tokens2[:ind2] + tokens2[ind2 + 1:]
            deleted = ["[DELETED]" + token for token in tokens1]
            added = ["[ADDED]" + token for token in tokens2]
            common = ["[FIXED]" + token for token in allCommon]
            return deleted + added  # +common

        assert len(generated) == len(promptQuestion) == 1
        generated = generated["sent"][0].split(" ")
        promptQuestion = promptQuestion["sent"][0].split(" ")
        prediction = _get_edits(promptQuestion, generated)
        edit_f1 = 0
        for _question in question["sent"]:
            _question = _question.split(" ")
            reference = _get_edits(promptQuestion, _question)
            # now compare the reference edits and predicted edits
            if len(reference) == len(prediction) == 0:
                # rarely, reference has no edits after normalization
                # then, if the prediction also has no edits, it gets full score
                edit_f1 = 1
            elif len(reference) == 0 or len(prediction) == 0:
                # if only one of them has no edits, zero score
                edit_f1 = max(edit_f1, 0)
            else:
                # otherwise, compute F1 score between prediction and reference
                edit_f1 = max(edit_f1, get_f1(prediction, reference, is_equal=lambda x, y: x == y))
        evaluation["edit-f1"] = edit_f1

    assert len(metrics) == len(evaluation)
    return evaluation


def get_exact_match(answers1, answers2):
    if type(answers1) == list:
        if len(answers1) == 0:
            return 0
        return np.max([get_exact_match(a, answers2) for a in answers1])
    if type(answers2) == list:
        if len(answers2) == 0:
            return 0
        return np.max([get_exact_match(answers1, a) for a in answers2])
    return (normalize_answer(answers1) == normalize_answer(answers2))


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(str(s)))))


def get_f1(answers, predictions, is_equal=get_exact_match, return_p_and_r=False, name='f1'):
    '''
    :answers: a list of list of strings
    :predictions: a list of strings
    '''
    assert len(answers) > 0 and len(predictions) > 0, (answers, predictions)
    occupied_answers = [False for _ in answers]
    occupied_predictions = [False for _ in predictions]
    for i, answer in enumerate(answers):
        for j, prediction in enumerate(predictions):
            if occupied_answers[i] or occupied_predictions[j]:
                continue
            em = is_equal(answer, prediction)
            if em:
                occupied_answers[i] = True
                occupied_predictions[j] = True
    assert np.sum(occupied_answers) == np.sum(occupied_predictions)
    a, b = np.mean(occupied_answers), np.mean(occupied_predictions)
    if return_p_and_r:
        if a + b == 0:
            return 0., 0., 0.
        return 2 * a * b / (a + b), float(a), float(b)
    if a + b == 0:
        return 0.
    if name == 'p':
        return float(b)
    elif name == 'r':
        return float(a)
    return 2 * a * b / (a + b)


def load_reference(reference_path):
    if os.path.exists(reference_path):
        with open(reference_path, "r") as f:
            reference = json.load(f)
        if not (type(reference) == list and \
                all([type(ref) == dict and "id" in ref and "question" in ref and "annotations" in ref and \
                     type(ref["question"]) == str and type(ref["annotations"]) == list and \
                     all([type(ann) == dict and ann["type"] in ["singleAnswer", "multipleQAs"] for ann in
                          ref["annotations"]]) \
                     for ref in reference])):
            raise Exception("Reference file {} is wrong".format(reference_path))
    else:
        raise Exception("Reference file {} not found".format(reference_path))
    return reference


def load_prediction(prediction_path, ids):
    if os.path.exists(prediction_path):
        with open(prediction_path, "r") as f:
            prediction = json.load(f)
        if str(list(prediction.keys())[0]) == int:
            prediction = {str(key): value for key, value in prediction.items()}
        if type(list(prediction.values())[0]) == str:
            prediction = {key: [value] for key, value in prediction.items()}
        if not (type(prediction) == dict and \
                len(ids - set(prediction.keys())) == 0):
            raise Exception("Prediction file {} is wrong".format(prediction_path))
        if not (all([type(pred) == list for pred in prediction.values()]) and \
                (all([type(p) == str for pred in prediction.values() for p in pred]) or \
                 all([type(p) == dict and "question" in p and "answer" in p \
                      and type(p["question"]) == type(p["answer"]) == str for pred in prediction.values() for p in
                      pred]))):
            raise Exception("Prediction file {} has a wrong format".format(prediction_path))
    else:
        raise Exception("Prediction file {} not found".format(prediction_path))
    return prediction


import csv

def load_data(file, collection='wiki-text'):
    csv.register_dialect('tsv_dialect', delimiter='\t', quoting=csv.QUOTE_ALL)
    data = []
    with open(file, "r") as wf:
        reader = csv.DictReader(wf, fieldnames=['id', 'text', 'title'], dialect='tsv_dialect')
        for row in reader:
            data.append(dict(row))
    csv.unregister_dialect('tsv_dialect')
    data = data[1:]
    print(len(data))
    print(data[0])
    data = [{'pid': i, 'id': item['id'], 'text': item['text'], 'title': item['title']} for i, item in enumerate(data)]
    passage = TextPassage(collection)
    print('start write')
    passage.write(data)
    print('write done')
    print(len(passage))


if __name__ == "__main__":
    load_data('data/wikipedia/psgs_w100.tsv')
