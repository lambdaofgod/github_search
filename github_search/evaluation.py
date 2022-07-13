from github_search import ir_utils, utils
import sentence_transformers

from typing import TypeVar, List, Optional
import torch
from github_search.types import Encoder

CorpusType = TypeVar("CorpusType")
QueryType = TypeVar("QueryType")


def get_ir_results(
    ir_evaluator,
    model: Encoder[CorpusType],
    corpus_representations: List[CorpusType],
    query_model: Optional[Encoder[QueryType]] = None,
):
    corpus_embeddings = model.encode(corpus_representations, convert_to_tensor=True)
    if query_model is not None:
        return ir_evaluator.compute_metrices(
            query_model, model, corpus_embeddings=corpus_embeddings
        )
    else:
        return ir_evaluator.compute_metrices(
            model, model, corpus_embeddings=corpus_embeddings
        )
