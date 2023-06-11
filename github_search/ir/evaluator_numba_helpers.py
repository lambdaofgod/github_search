import numba
import numba as nb
from numba import types
import numpy as np


@numba.jit(nopython=True)
def fast_containment(elem, docs):
    for doc in docs:
        if str(elem) == str(doc):
            return True
    return False


@numba.jit(nopython=True)
def fast_count(top_hit_corpus_ids, query_relevant_docs):
    total_hits = 0
    for hit_corpus_id in top_hit_corpus_ids:
        if fast_containment(hit_corpus_id, query_relevant_docs):
            total_hits += 1
    return total_hits


@numba.jit(nopython=True)
def compute_fast_mrr(top_hit_corpus_ids, query_relevant_docs):
    mrrs = np.zeros(len(top_hit_corpus_ids))
    for rank, hit in enumerate(top_hit_corpus_ids):
        if fast_containment(hit, query_relevant_docs):
            mrrs[rank] = 1.0 / (rank + 1)
            break
    return mrrs


@numba.jit(nopython=True)
def compute_fast_dcg_at_k(relevances, k):
    dcg = 0
    for i in range(min(len(relevances), k)):
        dcg += relevances[i] / np.log2(i + 2)  # +2 as we start our idx at 0
    return dcg
