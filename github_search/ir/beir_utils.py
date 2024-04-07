import pandas as pd
from typing import Dict, List, Tuple, Union
from pydantic import BaseModel
import pytrec_eval
from github_search.ir import ir_utils
import logging
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval.search.lexical import BM25Search as BM25
import math
from operator import itemgetter


Retriever = Union[BM25, DRES]


class IRData(BaseModel):
    queries: dict
    corpus: dict
    qrels: dict


class RetrievalResults(BaseModel):
    results: dict
    metrics: dict

    def get_metrics_df(self):
        return pd.DataFrame.from_records([self.metrics])


def load_ir_data(df, text_cols, max_words=1000):
    return IRData(
        queries=ir_utils.BEIRAdapter.get_queries(df, "true_tasks"),
        corpus=ir_utils.BEIRAdapter.get_corpus(
            df, "repo", text_cols, max_words=max_words
        ),
        qrels=ir_utils.BEIRAdapter.get_qrels(df, "repo", "true_tasks"),
    )


class PerQueryMetrics:
    @staticmethod
    def get_hits_at_k(
        qrels: Dict[str, Dict[str, int]],
        results: Dict[str, Dict[str, float]],
        k_values: List[int],
    ) -> Tuple[Dict[str, float]]:
        per_query_hits = {}

        k_max, top_hits = max(k_values), {}

        for query_id, doc_scores in results.items():
            top_hits[query_id] = [
                item[0]
                for item in sorted(
                    doc_scores.items(), key=lambda item: item[1], reverse=True
                )[0:k_max]
            ]

        for query_id in top_hits:
            hits = {}
            for k in k_values:
                hits[f"Hits@{k}"] = 0.0
            query_relevant_docs = set(
                [doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0]
            )
            for k in k_values:
                for relevant_doc_id in query_relevant_docs:
                    if relevant_doc_id in top_hits[query_id][0:k]:
                        hits[f"Hits@{k}"] += 1.0
            per_query_hits[query_id] = hits

        return per_query_hits

    @staticmethod
    def get_per_query_metrics_df(qrels, results, k_values):
        per_query_metrics_df = pd.DataFrame(
            PerQueryMetrics.get_hits_at_k(qrels, results, k_values)
        ).T
        for i in k_values:
            per_query_metrics_df[f"Accuracy@{i}"] = 1.0 * (
                per_query_metrics_df[f"Hits@{i}"] > 0
            )
        return per_query_metrics_df


class PerQueryIREvaluator(BaseModel):
    k_values: List[int]

    def get_scores(self, ir_data, retriever: Union[Retriever, List[Retriever]]):
        results = self.get_results(retriever, ir_data)

        map_string = "map_cut." + ",".join([str(k) for k in self.k_values])
        ndcg_string = "ndcg_cut." + ",".join([str(k) for k in self.k_values])
        recall_string = "recall." + ",".join([str(k) for k in self.k_values])
        precision_string = "P." + ",".join([str(k) for k in self.k_values])
        evaluator = pytrec_eval.RelevanceEvaluator(
            ir_data.qrels, {map_string, ndcg_string, recall_string, precision_string}
        )
        scores = evaluator.evaluate(results)
        return PerQueryMetrics.get_per_query_metrics_df(
            ir_data.qrels, results, self.k_values
        )

    def get_results(self, retriever: Union[Retriever, List[Retriever]], ir_data):
        if type(retriever) is list:
            multi_retriever_results = [
                r.retrieve(ir_data.corpus, ir_data.queries) for r in retriever
            ]
            return self._merge_results(multi_retriever_results)
        else:
            return retriever.retrieve(ir_data.corpus, ir_data.queries)

    @classmethod
    def _merge_results(cls, results):
        queries = results[0].keys()
        merged_results = {}
        for query in queries:
            query_results = cls._merge_query_results(
                results[0][query], results[1][query]
            )
            merged_results[query] = query_results
        return merged_results

    @classmethod
    def _merge_query_results(cls, d1, d2):
        total_len = max(len(d1), len(d2))
        half_merged_len = math.ceil(total_len / 2)
        merged_results = {}
        d1_items = cls._get_normalized_scores_items(d1)
        d2_items = cls._get_normalized_scores_items(d2)
        while len(merged_results) < total_len:
            if len(d1_items) > 0:
                retrieved, score = d1_items.pop()
                merged_results[retrieved] = score
            if len(d2_items) > 0:
                retrieved, score = d2_items.pop()
                merged_results[retrieved] = score
        return merged_results

    @classmethod
    def _get_normalized_scores_items(cls, query_results):
        total_score = sum(query_results.values())
        normalized_query_results = {
            res: score / total_score
            for (res, score) in sorted(query_results.items(), key=itemgetter(1))
        }
        return list(normalized_query_results.items())


class MultiTextEvaluator(BaseModel):
    """
    Evaluate a dataframe that has multiple texts for each query (multiple generation experiments)
    iteration_col says which experiment it was
    """

    iteration_col: str
    text_cols: List[str]
    k_values: List[int] = [1, 5, 10, 25]
    verbose: bool = True

    def get_ir_datas(self, df):
        for iter in df[self.iteration_col].unique():
            ir_data = load_ir_data(df[df[self.iteration_col] == iter], self.text_cols)
            yield (iter, ir_data)

    def evaluate(self, df, retriever):
        ir_datas = dict(self.get_ir_datas(df))
        dfs = []
        for iter, ir_data in ir_datas.items():
            if self.verbose:
                logging.info(f"Running iteration {iter}")
            per_query_evaluator = PerQueryIREvaluator(k_values=self.k_values)
            df = per_query_evaluator.get_scores(ir_data, retriever)
            df[self.iteration_col] = iter
            dfs.append(df)
        metrics_df = pd.concat(dfs)
        metrics_df["query"] = metrics_df.index
        return metrics_df
