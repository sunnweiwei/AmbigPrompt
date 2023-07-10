import os

import numpy as np
from tqdm import tqdm
from torch.nn import DataParallel
from tools import TextPassage
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import torch
import torch.nn.functional as F
import json
import faiss
import time


def mean_pooling(model_output, attention_mask, **kwargs):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    x = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return F.normalize(x, p=2, dim=1)


def build_index(collection, shard=True, dim=None, gpu=True):
    t = time.time()
    dim = collection.shape[1] if dim is None else dim
    cpu_index = faiss.index_factory(dim, "Flat", faiss.METRIC_INNER_PRODUCT)
    # cpu_index = faiss.index_factory(dim, 'OPQ32,IVF1024,PQ32')
    if gpu:
        ngpus = faiss.get_num_gpus()
        co = faiss.GpuMultipleClonerOptions()
        co.shard = shard
        gpu_index = faiss.index_cpu_to_all_gpus(cpu_index, co=co)
        index = gpu_index
    else:
        index = cpu_index
    # gpu_index.train(xb)
    index.add(collection)
    print(f'build index of {len(collection)} instances, time cost ={time.time() - t}')
    return index


def do_retrieval(xq, index, k=1):
    t = time.time()
    distance, rank = index.search(xq, k)
    print(f'search {len(xq)} queries, time cost ={time.time() - t}')
    return rank, distance


def cls_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    return token_embeddings[:, 0]


def encode_query(file):
    tokenizer = AutoTokenizer.from_pretrained('Luyu/co-condenser-wiki')
    model = AutoModel.from_pretrained('Luyu/co-condenser-wiki')
    model.load_state_dict(torch.load('out/dense/pytorch_model.bin', map_location=lambda storage, loc: storage))

    collect = []
    start, end = 0, 0
    batch_size = 512
    dev = json.load(open(file))
    query_embedding = np.zeros(shape=(len(dev), 768), dtype=np.float32)
    for i in tqdm(range(len(dev)), total=len(dev)):
        item = dev[i]
        sentence = f"{item['question']}"
        collect.append(sentence)
        if len(collect) == batch_size:
            encoded_input = tokenizer(collect, padding=True, truncation=True, return_tensors='pt')
            with torch.no_grad():
                model_output = model(**encoded_input)
                sentence_embeddings = cls_pooling(model_output, encoded_input['attention_mask'])
            end = start + len(sentence_embeddings)
            query_embedding[start:end, ] = sentence_embeddings
            start = end
            collect = []
            assert end == i + 1

    encoded_input = tokenizer(collect, padding=True, truncation=True, return_tensors='pt')
    encoded_input = {k: v.cuda() for k, v in encoded_input.items()}
    with torch.no_grad():
        model_output = model(**encoded_input)
        sentence_embeddings = cls_pooling(model_output, encoded_input['attention_mask'])
    end = start + len(sentence_embeddings)
    query_embedding[start:end, ] = sentence_embeddings
    start = end
    return query_embedding


def main(file):
    os.makedirs('out/dense/', exist_ok=True)
    queries = encode_query(file)
    passage = TextPassage()
    collection = np.memmap(f'out/dense/corpus.dat', dtype='float32', mode="r", shape=(len(passage), 768))
    index = build_index(collection, gpu=False)
    rank, distance = do_retrieval(queries, index, k=100)
    rank = rank.tolist()
    json.dump(rank, open(file + '.rank', 'w'))
    data = json.load(open(file))
    new_data = [{"question": item['question'], 'answers': item['answers'], 'passages':passages}
                for item, passages in zip(data, rank)]
    json.dump(new_data, open(file, 'w'))


if __name__ == '__main__':
    main(file='data/ambig/train.json')
