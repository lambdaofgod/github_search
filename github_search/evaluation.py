from github_search import ir_utils, utils
import sentence_transformers

from typing import Protocol, TypeVar, Generic, List, Optional
import torch

T = TypeVar("T")


class Encoder(Protocol, Generic[T]):
    """
    encoder that generalizes SentenceTransformers - Encoder[str]
    to encoders that can handle data with other types like graphs
    """

    def encode(
        inputs: List[T],
        show_progress_bar: bool = False,
        batch_size: int = 32,
        convert_to_tensor: bool = True,
    ) -> torch.Tensor:
        """encode raw data"""


CorpusType = TypeVar("CorpusType")
QueryType = TypeVar("QueryType")


def get_ir_results(
    ir_evaluator,
    model: Encoder[QueryType],
    corpus_representations: List[CorpusType],
    corpus_model: Optional[Encoder[CorpusType]] = None,
):
    corpus_embeddings = model.encode(corpus_representations, convert_to_tensor=True)
    return ir_evaluator.compute_metrices(
        model, corpus_model, corpus_embeddings=corpus_embeddings
    )
