import json
import logging

import itertools
import random
from typing import List, Union
import fire
import yaml
import pandas as pd
from beir import LoggingHandler
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25
from pydantic import BaseModel, Field
from zenml.client import Client
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from github_search.ir.beir_utils import MultiTextEvaluator, load_ir_data

# Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)


class EvaluationRunConfig(BaseModel):
    text_cols: List[str] = Field(default=["dependencies", "tasks"])
    retriever_name: Union[str, List[str]] = Field(default="bm25")
    k_values: List[int]


class EvaluationConfig(BaseModel):
    text_cols: List[List[str]]
    retriever_names: List[Union[str, List[str]]]
    k_values: List[int] = Field(default=[1, 5, 10, 25])

    def load_from_yaml(self, path):
        with open(path) as f:
            config = yaml.safe_load(f)
        return EvaluationConfig.parse_obj(config)

    @property
    def run_configs(self):
        for text_cols, retriever_name in itertools.product(
            self.text_cols, self.retriever_names
        ):
            yield EvaluationRunConfig(
                text_cols=text_cols,
                retriever_name=retriever_name,
                k_values=self.k_values,
            )


def get_bm25_retriever(k_values, index_name):
    hostname = "localhost"  # localhost

    #### Intialize ####
    # (1) True - Delete existing index and re-index all documents from scratch
    # (2) False - Load existing index
    initialize = True  # False
    number_of_shards = 1
    model = BM25(
        index_name=index_name,
        hostname=hostname,
        initialize=initialize,
        number_of_shards=number_of_shards,
    )

    # (2) For datasets with big corpus ==> keep default configuration
    # model = BM25(index_name=index_name, hostname=hostname, initialize=initialize)
    return EvaluateRetrieval(model, k_values=k_values)


def get_sentence_transformer_retriever(
    k_values, model_name="msmarco-distilbert-base-tas-b"
):
    model = DRES(models.SentenceBERT(model_name), batch_size=64)
    return EvaluateRetrieval(model, score_function="dot", k_values=k_values)


def get_retriever(retriever_name, k_values, index_name):
    if retriever_name == "bm25":
        return get_bm25_retriever(k_values, index_name)
    elif type(retriever_name) is list:
        return [get_retriever(rname, k_values, index_name) for rname in retriever_name]
    else:
        return get_sentence_transformer_retriever(k_values, retriever_name)


def get_ir_results(retriever, ir_data):
    #### Sharding ####
    # (1) For datasets with small corpus (datasets ~ < 5k docs) => limit shards = 1
    # SciFact is a relatively small dataset! (limit shards to 1)

    # Retrieve dense results (format of results is identical to qrels)
    results = retriever.retrieve(ir_data.corpus, ir_data.queries)

    # Evaluate your retrieval using NDCG@k, MAP@K ...
    logging.info("Retriever evaluation for k in: {}".format(
        retriever.k_values))

    ndcg, _map, recall, precision = retriever.evaluate(
        ir_data.qrels, results, retriever.k_values
    )
    accuracy = EvaluateRetrieval.evaluate_custom(
        ir_data.qrels, results, k_values=retriever.k_values, metric="accuracy@k"
    )
    return RetrievalResults(
        results=results, metrics={**accuracy, **
                                  ndcg, **_map, **recall, **precision}
    )


def print_ir_results_sample(ir_data, results):
    query_id, scores_dict = random.choice(list(ir_data.results.items()))
    logging.info("Query : %s\n" % ir_data.queries[query_id])

    scores = sorted(scores_dict.items(),
                    key=lambda item: item[1], reverse=True)
    for rank in range(10):
        doc_id = scores[rank][0]
        logging.info(
            "Doc %d: %s - %s\n"
            % (
                rank + 1,
                doc_id,
                ir_data.corpus[doc_id].get("true_tasks"),
            )
        )


class RetrievalConfig(BaseModel):
    text_cols: list = Field(default=[["dependencies", "tasks"]])
    retriever_models: list = Field(default=["bm25"])


def load_generation_metrics_df():
    # TODO delete this after merging the new ir evaluation with pipeline
    # this is search df from previous pipeline
    artifact = Client().get_artifact("a9bbe6aa-78bd-4eff-9958-ca03c6831caf")
    return artifact.load()


def get_retriever_sanitized_name(retriever_name):
    if type(retriever_name) is str:
        return retriever_name.replace("/", "_")
    else:
        return "-".join(rname.replace("/", "_") for rname in retriever_name)


def main(text_cols_str=None, text_col_config_path: str = None):
    generation_metrics_df = load_generation_metrics_df()
    if text_col_config_path is None:
        if text_cols_str is not None:
            text_cols = [[s.strip() for s in text_cols_str.split(",")]]
        else:
            text_cols = [["dependencies", "tasks"]]
        retrieval_config = RetrievalConfig(text_cols=text_cols)
    else:
        with open(text_col_config_path) as f:
            retrieval_config = RetrievalConfig.parse_obj(json.load(f))

    iterations = generation_metrics_df["generation"].unique().tolist()
    for retriever_model in retrieval_config.retriever_models:
        index_name = "github_search_{}"
        retriever = get_retriever(
            retriever_model, k_values=[1, 5, 10, 25], index_name=index_name
        )
        metrics_records = []
        for iteration in iterations:
            for cols in retrieval_config.text_cols:
                logging.info(f"Retriever: {retriever_model}")
                logging.info(f"Evaluating cols: {cols}")
                logging.info(f"Iteration: {iteration}")
                metrics = evaluate_iteration(
                    retriever, retriever_model, generation_metrics_df, iteration, cols
                )
                metrics_records.append(metrics)
        sanitized_retriever_model_name = get_retriever_sanitized_name(
            retriever_model)
        pd.DataFrame.from_records(metrics_records).to_json(
            f"output/beir/{sanitized_retriever_model_name}_metrics.json"
        )


def evaluate_run(search_df, run_config: EvaluationRunConfig):
    multi_evaluator = MultiTextEvaluator(
        iteration_col="generation", text_cols=run_config.text_cols
    )
    logging.info(f"Evaluating retriever: {run_config.retriever_name}")
    logging.info(f"Text columns: {run_config.text_cols}")
    index_name = f"github_search_{'_'.join(run_config.text_cols)}"
    retriever = get_retriever(
        run_config.retriever_name, k_values=run_config.k_values, index_name=index_name
    )
    per_query_metrics_df = multi_evaluator.evaluate(search_df, retriever)
    per_query_metrics_df["retriever_name"] = get_retriever_sanitized_name(
        run_config.retriever_name
    )
    per_query_metrics_df["text_cols"] = str(run_config.text_cols)
    return per_query_metrics_df


def multi_main(search_df_path, out_dir):
    search_df = pd.read_parquet(search_df_path)
    # csv_search_df = pd.read_csv("/tmp/search_df.csv", encoding="utf-8")
    retriever_names = [
        "bm25",
        "sentence-transformers/all-mpnet-base-v2",
        [
            "bm25",
            "sentence-transformers/all-mpnet-base-v2",
        ],
    ]

    evaluation_config = EvaluationConfig(
        text_cols=[
            ["readme"],
            ["tasks", "readme"],
            ["tasks", "dependencies"],
            ["dependencies"],
            ["tasks"],
        ],
        retriever_names=retriever_names,
        k_values=[1, 5, 10, 25],
    )

    all_metrics_df = pd.DataFrame()

    for run_config in evaluation_config.run_configs:
        retriever_name_part = get_retriever_sanitized_name(
            run_config.retriever_name)
        text_cols_part = "-".join(run_config.text_cols)
        out_path = (
            f"{out_dir}/per_query_{retriever_name_part}_{text_cols_part}_metrics.csv"
        )
        per_query_metrics_df = evaluate_run(search_df, run_config)
        logging.info(f"Saving per query metrics to {out_path}")
        per_query_metrics_df.to_csv(out_path, index=False)
        all_metrics_df = pd.concat([all_metrics_df, per_query_metrics_df])
    all_metrics_df.to_csv(f"{out_dir}/ir_metrics.csv", index=False)


if __name__ == "__main__":
    fire.Fire(multi_main)
