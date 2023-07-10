from modeling import replace_attention_forward
import torch
from transformers import AutoTokenizer, AdamW, T5ForConditionalGeneration, T5Config
from accelerate import Accelerator
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import json
from tqdm import tqdm
import numpy as np
import argparse
import csv
import os
from collections import defaultdict
from tools import TextPassage, FiDT5, get_f1


NUM_PASSAGES = 100
replace_attention_forward(NUM_PASSAGES + 1)


class FiD(T5ForConditionalGeneration):
    def _prepare_encoder_decoder_kwargs_for_generation(self, input_ids, model_kwargs, *args, **kwargs):
        if "encoder_outputs" not in model_kwargs:
            # retrieve encoder hidden states
            encoder = self.get_encoder()
            encoder_kwargs = {
                argument: value
                for argument, value in model_kwargs.items()
                if not (argument.startswith("decoder_") or argument.startswith("cross_attn"))
            }
            batch_size, num_psg, length = input_ids.size()
            input_ids = input_ids.view(batch_size * num_psg, length)
            encoder_kwargs['attention_mask'] = encoder_kwargs['attention_mask'].view(batch_size * num_psg, length)
            encoder_kwargs.pop('use_cache')
            encoder_outputs = encoder(input_ids=input_ids, return_dict=True, use_cache=False, **encoder_kwargs)
            encoder_outputs.last_hidden_state = encoder_outputs.last_hidden_state.view(batch_size, num_psg * length, -1)
            model_kwargs["encoder_outputs"] = encoder_outputs
            model_kwargs['attention_mask'] = model_kwargs['attention_mask'].view(batch_size, -1)
        return model_kwargs

    def forward(self, input_ids=None, attention_mask=None, encoder_outputs=None, *args, **kwargs):
        if encoder_outputs is None:
            batch_size, num_psg, length = input_ids.size()
            input_ids = input_ids.view(batch_size * num_psg, length)
            _attention_mask = attention_mask.view(batch_size * num_psg, length)
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=_attention_mask,
                head_mask=kwargs.get('head_mask', None),
                output_attentions=kwargs.get('output_attentions', True),
                output_hidden_states=kwargs.get('output_hidden_states', True),
                return_dict=True,
            )
            encoder_outputs.last_hidden_state = encoder_outputs.last_hidden_state.view(batch_size, num_psg * length, -1)
            attention_mask = attention_mask.view(batch_size, -1)

        return super().forward(
            input_ids=input_ids, attention_mask=attention_mask, encoder_outputs=encoder_outputs, *args, **kwargs)


class MultiQAData(Dataset):
    def __init__(self, data, passages, tokenizer, lamb=0.5, top_k=100, max_length=256, trunc_rate=0.5):
        self.data = data
        self.passages = passages
        self.tokenizer = tokenizer
        self.lamb = lamb
        self.top_k = top_k
        self.max_length = max_length
        self.trunc_rate = trunc_rate

    def __len__(self):
        return len(self.data)

    def get_item(self, question, passages, prefix, target):
        passages = passages + ["[PAD]"] * (self.top_k - len(passages))
        context = [f"Question: {question} ## Passage: {psg}" for psg in passages]

        if np.random.rand() < self.lamb:
            prefix = prefix + [target]
            target = "[EOI]"
        elif len(prefix) > 0 and np.random.rand() < self.trunc_rate:
            prefix = prefix[:np.random.randint(len(prefix))]

        prefix = f"Question: {question} ## Previous Answers: " + " ; ".join(prefix)

        inputs = [prefix] + context

        inputs = [torch.tensor(self.tokenizer.encode(x, truncation=True, max_length=self.max_length)) for x in inputs]
        target = torch.tensor(self.tokenizer.encode(target, truncation=True, max_length=32))
        return inputs, target

    def __getitem__(self, i):
        item = self.data[i]
        question = item['question'].lower().strip()
        passages = [self.passages[pid] for pid in item['passages']][:self.top_k]
        prefix = item['prefix']
        target = item['target']
        np.random.shuffle(prefix)
        return self.get_item(question, passages, prefix, target)

    def collate_fn(self, data):
        inputs, target = zip(*data)
        batch_size, num_psg = len(inputs), len(inputs[0])
        inputs = sum(inputs, [])  # batch_size * num_psg, length
        input_ids = pad_sequence(inputs, batch_first=True,
                                 padding_value=self.tokenizer.pad_token_id)  # batch_size * num_psg, length
        input_ids = input_ids.view(batch_size, num_psg, -1)  # batch_size, num_psg, length
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        labels = pad_sequence(target, batch_first=True, padding_value=-100)
        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )


def pre_data(data):
    prefix_target_data = []
    for item in data:
        question = item['question']
        answers = sorted(item['answers'], key=lambda x: -len(x))[0]
        answers = [a[0] for a in answers]
        passages = item['passages']
        for i in range(len(answers)):
            prefix_target_data.append({
                'question': question,
                'passages': passages,
                'prefix': answers[:i] + answers[i + 1:],
                'target': answers[i]
            })
    return prefix_target_data


def run(data_path, save_path):
    accelerator = Accelerator(gradient_accumulation_steps=4)

    epochs = 10
    batch_size = 1
    os.makedirs(save_path, exist_ok=True)

    print(save_path)

    config = T5Config.from_pretrained('t5-base')
    model = FiD(config)

    # Load FiD Checkpoint
    try:
        checkpoint_model = FiDT5.from_pretrained('fid/ckpt/nq_reader_base')
        checkpoint_model.unwrap_encoder()
        model.load_state_dict(checkpoint_model.state_dict(), strict=False)
    except:
        pass

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained('./t5-base')

    train_dataset = MultiQAData(
        data=pre_data(json.load(open(data_path))),
        passages=TextPassage('wiki-text'),
        tokenizer=tokenizer,
        top_k=NUM_PASSAGES, max_length=200)

    print(len(train_dataset))

    data_loader = torch.utils.data.DataLoader(
        train_dataset, collate_fn=train_dataset.collate_fn, batch_size=batch_size, shuffle=True, num_workers=8)

    optimizer = AdamW(model.parameters(), lr=1e-4)
    model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)

    for epoch in range(epochs):
        accelerator.wait_for_everyone()
        accelerator.print(f'train epoch={epoch}')
        model = model.cuda()
        model.train()
        tk0 = tqdm(data_loader, total=len(data_loader))
        losses = []
        for batch in tk0:
            with accelerator.accumulate(model):
                output = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels'],
                    return_dict=True
                )

                loss = output.loss
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                optimizer.zero_grad()
                losses.append(loss.item())
                tk0.set_postfix(loss=sum(losses) / len(losses))
        if accelerator.is_main_process:
            accelerator.save(accelerator.unwrap_model(model).state_dict(), f'{save_path}/{epoch}.pt')
        accelerator.wait_for_everyone()


def pre_test_data(data):
    test_data = []
    for item in data:
        test_data.append({
            'question': item['question'],
            'passages': item['passages'],
            'target': item['answers']
        })
    return test_data


def test(data_path, save_path):
    config = T5Config.from_pretrained('./t5-base')
    model = FiD(config)
    file = f'{save_path}'
    model.load_state_dict(torch.load(file))
    tokenizer = AutoTokenizer.from_pretrained('./t5-base')
    dev_dataset = MultiQAData(
        data=pre_test_data(json.load(open(data_path))),
        passages=TextPassage('wiki-text'),
        tokenizer=tokenizer,
        top_k=NUM_PASSAGES, max_length=32, lamb=0, trunc_rate=0)

    tk0 = tqdm(range(len(dev_dataset)), total=len(dev_dataset))
    idx = 0
    score = defaultdict(list)
    model = model.cuda()
    model.eval()

    results = []

    for batch_id in tk0:
        pred_answer = []
        eoi = False
        noi = 0
        while not eoi:
            noi += 1
            item = dev_dataset.data[batch_id]
            question = item['question'].lower().strip()
            passages = [dev_dataset.passages[pid] for pid in item['passages']][:dev_dataset.top_k]
            prefix = pred_answer
            target = "[None]"
            inputs, target = dev_dataset.get_item(question, passages, prefix, target)
            batch = dev_dataset.collate_fn([[inputs, target]])
            out = model.generate(
                input_ids=batch['input_ids'].to(model.device),
                attention_mask=batch['attention_mask'].to(model.device),
                max_length=32,
            )
            answer = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
            print(pred_answer)
            pred_answer.append(answer)
            if len(pred_answer) > 1 and pred_answer[-1] in pred_answer[-2]:
                pred_answer = pred_answer[:-1]
            pred_answer = list(set(pred_answer))
            if '[EOI]' in pred_answer or noi >= 10:
                eoi = True

        item = dev_dataset.data[idx]

        answers = [[[a for a in ans if a is not None] for ans in ann] for ann in item['target']]

        answer_num = max([len(ann) for ann in answers])
        prediction1 = [text.strip() for text in pred_answer]
        prediction2 = list(set(prediction1))

        results.append(prediction2)

        f1s = np.max([get_f1(answer, prediction1) for answer in answers])
        f1s_wo_dupli = [get_f1(answer, prediction2, return_p_and_r=True) for answer in answers]
        f1s_wo_dupli.sort(key=lambda x: x[0], reverse=True)
        f1s_wo_dupli, rr, pp = f1s_wo_dupli[0]

        score['f1s'].append(f1s)
        score['f1s_wo'].append(f1s_wo_dupli)
        score['pp'].append(pp)
        score['rr'].append(rr)
        score['fid_len'].append(len(pred_answer))
        score['wo_len'].append(len(prediction2))
        score['true_len'].append(answer_num)
        if answer_num > 1:
            score['multi'].append(f1s_wo_dupli)
            idx += 1
        tk0.set_postfix(**{k: sum(v) / len(v) * 100 for k, v in score.items()})

    print({k: sum(v) / len(v) * 100 for k, v in score.items()})
    json.dump(results, open(f'{file}.results', 'w'))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/ambig/train.json')
    parser.add_argument('--save_path', type=str, default='out/ambig/model')
    parser.add_argument('--checkpoint', type=str, default='out/ambig/model/9.pt')
    parser.add_argument('--do_train', type=bool, default=True)
    parser.add_argument('--do_eval', type=bool, default=True)
    args = parser.parse_args()

    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))
    return args


if __name__ == '__main__':
    args = parse_args()
    if args.do_train:
        run(data_path=args.data_path, save_path=args.save_path)
    if args.de_eval:
        test(data_path=args.data_path, save_path=args.checkpoint)
