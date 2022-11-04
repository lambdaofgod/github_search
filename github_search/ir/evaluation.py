from typing import Generic, List, Optional, Protocol, TypeVar

import sentence_transformers
import torch
from github_search import utils
from github_search.ir import evaluator, ir_utils

CorpusType = TypeVar("CorpusType")
QueryType = TypeVar("QueryType")


def get_ir_results(
    ir_evaluator: sentence_transformers.evaluation.InformationRetrievalEvaluator,
    model: evaluator.Encoder[CorpusType],
    corpus_representations: List[CorpusType],
    query_model: Optional[evaluator.Encoder[QueryType]] = None,
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
