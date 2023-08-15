import ast
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd
from github_search import utils
from github_search.ir import (
    evaluator_impl,
    InformationRetrievalEvaluatorConfig,
    EmbedderPairConfig,
    InformationRetrievalColumnConfig,
)
from github_search.ir.models import EmbedderPair
from github_search.ir.ir_data import DictIRData


def round_float_dict(d, rounding=3):
    if type(d) is dict:
        return {k: round_float_dict(v, rounding) for k, v in d.items()}
    else:
        return float(round(d, rounding))


class InformationRetrievalEvaluator:
    """
    evaluate information retrieval metrics on unseen data

    this class essentially is refactored version of evaluator from sentence-transformers
    """

    doc_col = "document"

    def __init__(
        self, embedder_pair: EmbedderPair, rounding=4, use_test_time_augmentation=False
    ):
        self.document_embedder = embedder_pair.document_embedder
        self.query_embedder = embedder_pair.query_embedder
        self.rounding = rounding
        self.use_test_time_augmentation = use_test_time_augmentation

    def setup(self, df, column_config: InformationRetrievalColumnConfig):
        """
        prepare search dataset from `df`
        each row in `df` is a corpus object (document)
        each entry in `query_col` is assumed to be a list of queries for a document
        `document_col` will be used for retrieval
        """
        self.df = self.prepare_search_df(df, column_config).reset_index(drop=True)
        ir_evaluator = self._get_ir_evaluator_impl(
            self.df, query_col=column_config.query_col, doc_col=self.doc_col
        )
        self.query_col = column_config.query_col
        self.document_cols = column_config.document_cols
        self._ir_evaluator_impl = ir_evaluator

    @classmethod
    def prepare_search_df(cls, df, column_config: InformationRetrievalColumnConfig):
        search_df = utils.concatenate_flattened_list_cols(
            df,
            str_list_cols=column_config.list_cols,
            concat_cols=column_config.document_cols,
            target_col=InformationRetrievalEvaluator.doc_col,
        )
        query_col = column_config.query_col
        query_type = type(search_df[query_col].iloc[0])
        if query_type is str:
            return search_df.assign(
                **{query_col: search_df[query_col].apply(ast.literal_eval)}
            )
        elif query_type is np.ndarray:
            return search_df.assign(
                **{query_col: search_df[query_col].apply(lambda a: a.tolist())}
            )
        else:
            assert (
                query_type is list
            ), f"column {query_col} unsupported query type: {query_type}"
            return search_df

    @staticmethod
    def setup_from_df(
        search_df: pd.DataFrame, ir_config: InformationRetrievalEvaluatorConfig
    ):
        embedder_pair = EmbedderPair.from_config(ir_config.embedder_config)
        ir_evaluator = InformationRetrievalEvaluator(embedder_pair)
        ir_evaluator.setup(
            search_df,
            ir_config.column_config,
        )
        return ir_evaluator

    @staticmethod
    def setup_from_config(ir_config: InformationRetrievalEvaluatorConfig):
        raw_search_df = utils.pd_read_star(ir_config.search_df_path)
        return InformationRetrievalEvaluator.setup_from_df(raw_search_df, ir_config)

    def evaluate(
        self, queries_result_list=None, score_function_name="cos_sim", rounding=3
    ):
        if queries_result_list is None:
            queries_result_list = self.get_results_list()
        return self._ir_evaluator_impl.get_metrics_from_result_lists(
            queries_result_list,
            rounding=rounding,
        )[score_function_name]

    def get_results_list(self):
        corpus_representations = self.df[self.doc_col].to_list()
        corpus_embeddings = self.document_embedder.encode(
            corpus_representations, convert_to_tensor=True
        )
        return self._ir_evaluator_impl.get_queries_result_lists(
            self.query_embedder,
            self.document_embedder,
            corpus_embeddings=corpus_embeddings,
        )

    @classmethod
    def _get_ir_evaluator_impl(cls, df, query_col="tasks", doc_col="doc"):
        """
        the CustomInformationRetrievalEvaluatorImpl does the heavy lifting and is refactored
        but still its interface is too complex so we're hiding this complexity here
        """
        ir_dicts = DictIRData.from_pandas(
            df.dropna(subset=[query_col, doc_col]).reset_index(drop=True),
            query_col,
            doc_col,
        )
        ir_evaluator = evaluator_impl.CustomInformationRetrievalEvaluatorImpl(
            queries=ir_dicts.queries,
            corpus=ir_dicts.corpus,
            relevant_docs=ir_dicts.relevant_docs,
            main_score_function="cos_sim",
            map_at_k=[50],
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
        self,
        result_lists,
        used_similarity_metric="cos_sim",
    ) -> Dict[str, pd.DataFrame]:
        names = self.ir_evaluator.df[self.name_col]
        if result_lists is None:
            result_lists = (
                self.ir_evaluator._ir_evaluator_impl.get_queries_result_lists(
                    self.ir_evaluator.query_embedder,
                    self.ir_evaluator.document_embedder,
                )
            )

        return self.ir_evaluator._get_retrieval_prediction_dfs(
            self.ir_evaluator._ir_evaluator_impl.get_queries_dict(),
            list(names),
            result_lists[used_similarity_metric],
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
