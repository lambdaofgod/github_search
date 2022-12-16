from dataclasses import field, dataclass
import sentence_transformers

from github_search.utils import kwargs_only

@kwargs_only
@dataclass
class EmbedderPairConfig:

    query_embedder_path: str
    document_embedder_path: str
    doc_max_length: int = field(default=5000)
    query_max_length: int = field(default=100)


@kwargs_only
@dataclass
class EmbedderPair:

    query_embedder: sentence_transformers.SentenceTransformer
    document_embedder: sentence_transformers.SentenceTransformer

    @staticmethod
    def from_config(pair_config: EmbedderPairConfig):
        query_embedder = sentence_transformers.SentenceTransformer(
            pair_config.query_embedder_path
        )
        document_embedder = sentence_transformers.SentenceTransformer(
            pair_config.document_embedder_path
        )
        document_embedder.max_seq_length = pair_config.doc_max_length
        query_embedder.max_seq_length = pair_config.query_max_length
        return EmbedderPair(
            query_embedder=query_embedder, document_embedder=document_embedder
        )
