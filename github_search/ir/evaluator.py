import ast
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
import sentence_transformers
from github_search import utils
from github_search.ir import evaluator_impl
from github_search.ir.models import EmbedderPair, EmbedderPairConfig


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
class InformationRetrievalEvaluatorConfig:

    search_df_path: str
    document_col: str
    query_col: str
    embedder_config: EmbedderPairConfig


@dataclass
class InformationRetrievalEvaluator:

    """
    evaluate information retrieval metrics on unseen data
    """

    def __init__(
        self,
        embedder_pair: EmbedderPair,
        rounding=6,
    ):
        self.document_embedder = embedder_pair.document_embedder
        self.query_embedder = embedder_pair.query_embedder
        self.rounding = rounding

    def setup(self, df, query_col: str, document_col: str):
        """
        prepare search dataset from `df`
        each row in `df` is a corpus object (document)
        each entry in `query_col` is assumed to be a list of queries for a document
        `document_col` will be used for retrieval
        """
        ir_evaluator = self._get_ir_evaluator_impl(
            df, query_col=query_col, doc_col=document_col
        )
        self.query_col = query_col
        self.document_col = document_col
        self.df = df
        self._ir_evaluator_impl = ir_evaluator

    @staticmethod
    def setup_from_config(ir_config: InformationRetrievalEvaluatorConfig):
        search_df = utils.pd_read_star(ir_config.search_df_path)
        search_df[ir_config.query_col] = search_df[ir_config.query_col].apply(
            ast.literal_eval
        )
        embedder_pair = EmbedderPair.from_config(ir_config.embedder_config)
        ir_evaluator = InformationRetrievalEvaluator(embedder_pair)
        ir_evaluator.setup(search_df, ir_config.query_col, ir_config.document_col)
        return ir_evaluator

    def evaluate(self):
        return round_float_dict(
            self.get_ir_results(self.df[self.document_col]), self.rounding
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


@dataclass
class InformationRetrievalPredictor:

    """
    `name_col` will be displayed downstream
    """

    ir_evaluator: InformationRetrievalEvaluator
    name_col: str

    def get_predicted_documents(
        self, used_similarity_metric="cos_sim"
    ) -> Dict[str, pd.DataFrame]:
        names = self.ir_evaluator.df[self.name_col]
        result_lists = self.ir_evaluator._ir_evaluator_impl.get_queries_result_lists(
            self.ir_evaluator.query_embedder, self.ir_evaluator.document_embedder
        )[used_similarity_metric]
        return self.ir_evaluator._get_retrieval_prediction_dfs(
            self.ir_evaluator._ir_evaluator_impl.get_queries_dict(),
            list(names),
            result_lists,
        )

    def get_best_worst_results_df(self, tasks_with_areas_df, k=10):
        """
        best and worst queries with respect to hits@k
        """
        predicted_results = self.get_predicted_documents()

        hits_at_10 = self._get_per_query_hits_at_k(predicted_results, k=k)

        raw_tasks_with_hits_df = (
            tasks_with_areas_df.merge(
                pd.Series(hits_at_10, name="hits"), left_on="task", right_index=True
            )
            .drop(columns=["task_description"])
            .drop_duplicates()
        )

        best_tasks_with_hits_df = self._get_extremal_tasks_with_hits(
            raw_tasks_with_hits_df
        )
        worst_tasks_with_hits_df = self._get_extremal_tasks_with_hits(
            raw_tasks_with_hits_df, get_best=False
        )
        return {
            "best_tasks": best_tasks_with_hits_df,
            "worst_tasks": worst_tasks_with_hits_df,
        }

    @classmethod
    def _get_extremal_tasks_with_hits(
        cls, raw_tasks_with_hits_df, get_best=True, n_tasks_per_area=10
    ):
        """
        get best/worst performing tasks according to hits metric
        """
        tasks_with_hits_df = raw_tasks_with_hits_df.groupby("area").apply(
            lambda df: df.sort_values("hits", ascending=not get_best).iloc[
                :n_tasks_per_area
            ]
        )
        tasks_with_hits_df.index = tasks_with_hits_df.index.get_level_values("area")
        tasks_with_hits_df = tasks_with_hits_df.drop(columns="area")
        return tasks_with_hits_df

    def _get_per_query_hits_at_k(self, predicted_documents, k=10):
        """
        compare predicted documents to gold standards relevant docs
        """
        relevant_docs = self.ir_evaluator._ir_evaluator_impl.relevant_docs
        queries = self.ir_evaluator._ir_evaluator_impl.get_queries_dict()
        return {
            q: int(
                k
                * predicted_documents[q]
                .sort_values("score", ascending=False)
                .iloc[:k]["corpus_id"]
                .isin(relevant_docs[q_id])
                .mean()
            )
            for (q_id, q) in queries.items()
        }
