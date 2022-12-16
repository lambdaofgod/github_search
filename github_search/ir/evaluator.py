from dataclasses import dataclass
from typing import List, Dict, Optional

from github_search.ir import evaluator_impl
from github_search.ir.models import EmbedderPair
import sentence_transformers

import pandas as pd


def get_ir_dicts(input_df, query_col="tasks", doc_col="readme"):
    df_copy = input_df.copy()
    queries = df_copy[query_col].explode().drop_duplicates()
    queries = pd.DataFrame(
        {"query": queries, "query_id": [str(s) for s in queries.index]}
    )
    queries.index = queries["query_id"]
    corpus = df_copy[doc_col]
    corpus.index = [str(i) for i in corpus.index]
    df_copy["doc_id"] = corpus.index
    relevant_docs_str = df_copy[["doc_id", "tasks", doc_col]].explode(column="tasks")
    relevant_docs = (
        relevant_docs_str.merge(queries, left_on="tasks", right_on="query")[
            ["doc_id", "query_id"]
        ]
        .groupby("query_id")
        .apply(lambda df: set(df["doc_id"]))
        .to_dict()
    )
    return {
        "queries": queries["query"].to_dict(),
        "corpus": corpus.to_dict(),
        "relevant_docs": relevant_docs,
    }


def round_float_dict(d, rounding=3):
    if type(d) is dict:
        return {k: round_float_dict(v) for k, v in d.items()}
    else:
        return float(round(d, rounding))


@dataclass
class InformationRetrievalEvaluator:
    def __init__(
        self,
        embedder_pair: EmbedderPair,
        rounding=6,
    ):
        self.document_embedder = embedder_pair.document_embedder
        self.query_embedder = embedder_pair.query_embedder
        self.rounding = rounding

    def setup(
        self, df, query_col: str, document_col: str, name_col: Optional[str] = None
    ):
        if name_col is None:
            name_col = document_col

        ir_evaluator = self._get_ir_evaluator_impl(
            df, query_col=query_col, doc_col=document_col
        )
        self.name_col = name_col
        self.query_col = query_col
        self.document_col = document_col
        self.df = df
        self._ir_evaluator_impl = ir_evaluator

    def evaluate(self):
        return round_float_dict(
            self.get_ir_results(self.df[self.document_col]), self.rounding
        )

    @classmethod
    def _get_retrieval_prediction_dfs(
        cls,
        queries_dict: Dict[str, str],
        names: List[str],
        queries_result_list: List[List[dict]],
    ):
        return {
            queries_dict[i]: pd.DataFrame.from_records(
                [
                    {
                        "corpus_id": res_dict["corpus_id"],
                        "result": names[int(res_dict["corpus_id"])],
                        "score": res_dict["score"],
                    }
                    for res_dict in results
                ]
            )
            for (i, results) in zip(queries_dict.keys(), queries_result_list)
        }

    def get_predicted_documents(self, used_similarity_metric="cos_sim") -> Dict[str, pd.DataFrame]:
        names = self.df[self.name_col]
        result_lists = self._ir_evaluator_impl.get_queries_result_lists(
            self.query_embedder, self.document_embedder
        )[used_similarity_metric]
        return self._get_retrieval_prediction_dfs(
            self._ir_evaluator_impl.get_queries_dict(), list(names), result_lists
        )

    def get_ir_results(
        self,
        corpus_representations: List[str],
    ):
        corpus_embeddings = self.document_embedder.encode(
            corpus_representations, convert_to_tensor=True
        )
        return self._ir_evaluator_impl.compute_metrices(
            self.query_embedder,
            self.document_embedder,
            corpus_embeddings=corpus_embeddings,
        )

    @classmethod
    def _get_ir_evaluator_impl(cls, df, query_col="tasks", doc_col="readme"):
        ir_dicts = get_ir_dicts(
            df.dropna(subset=[query_col, doc_col]), query_col, doc_col
        )
        ir_evaluator = evaluator_impl.CustomInformationRetrievalEvaluatorImpl(
            **ir_dicts,
            main_score_function="cos_sim",
            map_at_k=[10],
            corpus_chunk_size=5000,
        )
        return ir_evaluator
