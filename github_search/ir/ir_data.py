from pydantic import BaseModel, Field
from typing import Set, Optional, Dict
import abc
import pandas as pd


class RelevanceResult(BaseModel):
    doc_id: str
    relevance_score: float = Field(default=1)


class QueryResult(BaseModel):
    query: str
    query_id: Optional[str]
    relevance_results: Set[RelevanceResult]


class IRDataBase(abc.ABC):
    @abc.abstractmethod
    def get_query(self, query_id: str) -> str:
        pass

    @abc.abstractmethod
    def get_doc(self, doc_id: str) -> str:
        pass

    @abc.abstractmethod
    def get_relevant_docs(self, query_id: str) -> Set[str]:
        pass


class DictIRData(IRDataBase, BaseModel):
    queries: Dict[str, str]
    corpus: Dict[str, str]
    relevant_docs: Dict[str, Set[str]]

    def get_query(self, query_id: str) -> str:
        return self.queries[query_id]

    def get_doc(self, doc_id: str) -> str:
        return self.corpus[doc_id]

    def get_relevant_docs(self, query_id: str) -> Set[str]:
        return self.relevant_docs[query_id]

    @staticmethod
    def from_pandas(input_df, query_col, doc_col):
        df_copy = input_df.copy()
        queries = df_copy[query_col].explode().drop_duplicates().reset_index(drop=True)
        queries = pd.DataFrame(
            {"query": queries, "query_id": [str(s) for s in queries.index]}
        )
        queries.index = queries["query_id"]
        corpus = df_copy[doc_col]
        corpus.index = [str(i) for i in corpus.index]
        df_copy["doc_id"] = corpus.index
        relevant_docs_str = df_copy[["doc_id", query_col, doc_col]].explode(
            column=query_col
        )
        relevant_docs = (
            relevant_docs_str.merge(queries, left_on=query_col, right_on="query")[
                ["doc_id", "query_id"]
            ]
            .groupby("query_id")
            .apply(lambda df: set(df["doc_id"]))
            .to_dict()
        )

        return DictIRData(
            **{
                "queries": queries["query"].to_dict(),
                "corpus": corpus.to_dict(),
                "relevant_docs": relevant_docs,
            }
        )
