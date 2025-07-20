from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.custom_metrics import mrr, recall_cap, hole, top_k_accuracy
from typing import Dict, List, Tuple
import logging
from typing import Union
import ast
from pydantic import BaseModel
import pandas as pd
from pathlib import Path
import math
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval.search.lexical import BM25Search as BM25
import pandas as pd
from typing import Dict, List, Tuple, Union
from pydantic import BaseModel
import pytrec_eval
import logging
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval.search.lexical import BM25Search as BM25
import math
from operator import itemgetter


Retriever = Union[BM25, DRES]


class CorpusDataLoader(BaseModel):
    repos_df_path: Union[str, Path]
    generated_readmes_df_path: Union[str, Path]
    code_df_path: Union[str, Path]

    @classmethod
    def from_dir(cls, dir):
        dir = Path(dir)
        return CorpusDataLoader(
            repos_df_path=dir / "sampled_repos.jsonl",
            generated_readmes_df_path=dir / "generated_readmes.jsonl",
            code_df_path=dir.parent.parent
            / "code"
            / "python_files_with_selected_code.feather",
        )

    def load_repos_df(self):
        assert self.repos_df_path.exists()
        df = pd.read_json(self.repos_df_path, orient="records", lines=True)
        if type(df["tasks"].iloc[0]) is str:
            df["tasks"] = df["tasks"].apply(ast.literal_eval)
        for col in ["repo", "tasks", "readme"]:
            assert col in df.columns
        return df

    def load_generated_readmes_df(self):
        assert self.generated_readmes_df_path.exists()
        if ".json" in str(self.generated_readmes_df_path):
            return self.load_generated_readmes_from_json()
        else:
            return self.load_generated_readmes_from_phoenix(
                self.generated_readmes_df_path
            )

    def load_generated_readmes_from_json(self):
        df = pd.read_json(self.generated_readmes_df_path, orient="records", lines=True)
        for col in ["rationale", "answer", "context_history", "repo_name"]:
            assert col in df.columns
        return df

    def load_python_code_df(self):
        assert self.code_df_path.exists()
        df = pd.read_feather(self.code_df_path)
        for col in ["content", "path", "repo_name", "tasks", "selected_code"]:
            assert col in df.columns
        return df

    def load_corpus_dfs(self, selected_repos=None):
        readme_df = self.load_repos_df()
        generated_readme_df = self.load_generated_readmes_df()
        selected_python_code_df = self.load_python_code_df()
        repos = set(readme_df["repo"]).intersection(
            set(generated_readme_df["repo_name"])
        )
        if selected_repos is not None:
            repos = repos.intersection(set(selected_repos))
        readme_df = readme_df[readme_df["repo"].isin(repos)].reset_index()
        generated_readme_df = (
            generated_readme_df.set_index("repo_name")
            .loc[readme_df["repo"]]
            .reset_index()
        )
        selected_python_code_df = selected_python_code_df[
            selected_python_code_df["repo_name"].isin(repos)
        ]
        return readme_df, generated_readme_df, selected_python_code_df

    @classmethod
    def load_generated_readmes_from_phoenix(cls, path):
        phoenix_trace_df = pd.read_parquet(path)
        phoenix_trace_df = phoenix_trace_df[
            (phoenix_trace_df["status_code"] == "OK")
            & (phoenix_trace_df["name"] == "Code2Documentation.forward")
        ]
        trace_generated_readmes_df = pd.json_normalize(
            phoenix_trace_df[phoenix_trace_df["name"] == "Code2Documentation.forward"][
                "attributes.output.value"
            ].apply(json.loads)
        )
        generated_readmes_df = pd.concat(
            [
                pd.json_normalize(
                    phoenix_trace_df["attributes.input.value"].apply(json.loads)
                ),
                trace_generated_readmes_df,
            ],
            axis=1,
        )
        return generated_readmes_df


def hits_at_k(
    qrels: Dict[str, Dict[str, int]],
    results: Dict[str, Dict[str, float]],
    k_values: List[int],
) -> Tuple[Dict[str, float]]:
    top_k_hits = {}

    for k in k_values:
        top_k_hits[f"Hits@{k}"] = 1

    k_max, top_hits = max(k_values), {}
    logging.info("\n")

    for query_id, doc_scores in results.items():
        top_hits[query_id] = [
            item[0]
            for item in sorted(
                doc_scores.items(), key=lambda item: item[1], reverse=True
            )[0:k_max]
        ]

    for query_id in top_hits:
        query_relevant_docs = set(
            [doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0]
        )
        for k in k_values:
            for relevant_doc_id in query_relevant_docs:
                if relevant_doc_id in top_hits[query_id][0:k]:
                    top_k_hits[f"Hits@{k}"] += 1

    for k in k_values:
        top_k_hits[f"Hits@{k}"] = round(top_k_hits[f"Hits@{k}"] / len(qrels), 5)
        logging.info("Hits@{}: {:.4f}".format(k, top_k_hits[f"Hits@{k}"]))

    return top_k_hits


class EvaluateRetrievalCustom(EvaluateRetrieval):
    @classmethod
    def evaluate_custom_multi(
        cls,
        qrels: Dict[str, Dict[str, int]],
        results: Dict[str, Dict[str, float]],
        k_values: List[int],
        metrics: List[str],
    ) -> Tuple[Dict[str, float]]:
        metric_values = dict()
        for metric in metrics:
            metric_values = metric_values | cls.evaluate_custom(
                qrels, results, k_values, metric
            )
        return metric_values

    @staticmethod
    def evaluate_custom(
        qrels: Dict[str, Dict[str, int]],
        results: Dict[str, Dict[str, float]],
        k_values: List[int],
        metric: str,
    ) -> Tuple[Dict[str, float]]:
        if metric.lower() in ["mrr", "mrr@k", "mrr_cut"]:
            return mrr(qrels, results, k_values)

        elif metric.lower() in ["hits@k"]:
            return hits_at_k(qrels, results, k_values)

        elif metric.lower() in ["recall_cap", "r_cap", "r_cap@k"]:
            return recall_cap(qrels, results, k_values)

        elif metric.lower() in ["hole", "hole@k"]:
            return hole(qrels, results, k_values)

        elif metric.lower() in [
            "acc",
            "top_k_acc",
            "accuracy",
            "accuracy@k",
            "top_k_accuracy",
        ]:
            return top_k_accuracy(qrels, results, k_values)


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
            per_query_metrics_df[f"Precision@{i}"] = 1.0 * (
                per_query_metrics_df[f"Hits@{i}"] / i
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
        scores_df = PerQueryMetrics.get_per_query_metrics_df(
            ir_data.qrels, results, self.k_values
        )
        queries_df = pd.DataFrame(
            {"query": ir_data.queries.values()}, index=ir_data.queries.keys()
        )
        return scores_df.merge(
            queries_df, left_index=True, right_index=True, how="outer"
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
