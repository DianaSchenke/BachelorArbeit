#source: https://huggingface.co/blog/g-ronimo/semscore
#extra file to cache EmbeddingModelWrapper

import torch
import re
from utils.semscore import EmbeddingModelWrapper
import torch.nn as nn
em = EmbeddingModelWrapper(bs=None)

def split_sentences(str):
    split_strs = re.split(r"[.?!:][\s\n]*", str)
    assert(split_strs)
    filtered_split_strs = [str for str in split_strs if len(str) > 10] #throw out strs that prbly aren't sentences
    if not filtered_split_strs:
        return split_strs #avoid problems in weird edgecases where response is REALLY short
    else:
        return  filtered_split_strs

def get_sem_score(predictions, reference):
    split_predictions = split_sentences(predictions)
    split_reference = split_sentences(reference)
    x = em.get_embeddings(split_predictions)
    x_hat = em.get_embeddings(split_reference)
    cos_sim = nn.CosineSimilarity(dim=1)
    tmp = []
    for i in range(x.size(0)):
        tmp.append(cos_sim(x[i], x_hat))
    sim_matrix = torch.stack(tmp)
    Recall = torch.mean(torch.max(sim_matrix, dim=1)[0])
    Precision = torch.mean(torch.max(sim_matrix, dim=0)[0])
    F1 = 2 * (Precision*Recall) / (Precision+Recall)
    return Recall.item(), Precision.item(), F1.item()
