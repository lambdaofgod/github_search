import math

import sentence_transformers
import torch.nn.functional as F


def setup_transformer():
    return sentence_transformers.SentenceTransformer(
        "flax-sentence-embeddings/st-codesearch-distilroberta-base"
    )


def attention(query, key, value, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(d_k)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return p_attn.matmul(value), p_attn


def global_attention(query, key, value, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(d_k)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return p_attn.matmul(value), p_attn
