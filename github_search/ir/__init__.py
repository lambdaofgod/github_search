from typing import List, Optional, Tuple

from github_search.utils import kwargs_only
from pydantic import BaseModel, Field


class EmbedderPairConfig(BaseModel, frozen=True):
    query_embedder_path: str
    document_embedder_path: str
    doc_max_length: Optional[int] = Field(default=5000)
    query_max_length: Optional[int] = Field(default=100)


class InformationRetrievalColumnConfig(BaseModel, frozen=True):
    document_cols: Tuple[str, ...]
    query_col: str
    list_cols: Tuple[str, ...]

    def select_columns(self, df):
        cols = df.columns
        return df[list(self.document_cols) + [self.query_col] + list(self.list_cols)]


class InformationRetrievalEvaluatorConfig(BaseModel, frozen=True):
    embedder_config: EmbedderPairConfig
    column_config: InformationRetrievalColumnConfig
