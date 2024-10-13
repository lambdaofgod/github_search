from dataclasses import field, dataclass
import sentence_transformers
from github_search import ir
from pydantic import BaseModel
import pandas as pd


@dataclass
class EmbedderPair:

    query_embedder: sentence_transformers.SentenceTransformer
    document_embedder: sentence_transformers.SentenceTransformer

    def init(
        self,
        *,
        query_embedder: sentence_transformers.SentenceTransformer,
        document_embedder: sentence_transformers.SentenceTransformer,
    ):
        return EmbedderPair(
            query_embedder=query_embedder, document_embedder=document_embedder
        )

    @staticmethod
    def from_config(pair_config: ir.EmbedderPairConfig):
        query_embedder = sentence_transformers.SentenceTransformer(
            pair_config.query_embedder_path
        )
        document_embedder = sentence_transformers.SentenceTransformer(
            pair_config.document_embedder_path
        )
        if pair_config.doc_max_length is not None:
            document_embedder.max_seq_length = pair_config.doc_max_length
            query_embedder.max_seq_length = pair_config.query_max_length
        return EmbedderPair(
            query_embedder=query_embedder, document_embedder=document_embedder
        )


class InformationRetrievalMetricsResult(BaseModel):
    per_query_metrics: pd.DataFrame
    aggregate_metrics: pd.DataFrame

    class Config:
        arbitrary_types_allowed = True

    def round(self, rounding=3):
        return InformationRetrievalMetricsResult(
            per_query_metrics=self.per_query_metrics.round(rounding),
            aggregate_metrics=self.aggregate_metrics.round(rounding),
        )
