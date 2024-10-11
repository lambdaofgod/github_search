from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.custom_metrics import mrr, recall_cap, hole, top_k_accuracy
from typing import Dict, List, Tuple
import logging
from typing import Union
import ast
from pydantic import BaseModel
import pandas as pd
from pathlib import Path


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
