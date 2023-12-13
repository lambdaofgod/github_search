import json
import logging

import random
import fire
import pandas as pd
from beir import LoggingHandler
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25
from github_search.ir import ir_utils
from pydantic import BaseModel, Field
from zenml.client import Client

artifact = Client().get_artifact("0c719bc2-cb21-4d26-8096-66d6a0cd1a1b")
generation_metrics_df = artifact.load()

# Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)


class IRData(BaseModel):
    queries: dict
    corpus: dict
    qrels: dict


class RetrievalResults(BaseModel):
    results: dict
    metrics: dict

    def get_metrics_df(self):
        return pd.DataFrame.from_records([self.metrics])


text_cols = ["dependencies", "tasks"]


def load_ir_data(df, text_cols):
    return IRData(
        queries=ir_utils.BEIRAdapter.get_queries(df, "true_tasks"),
        corpus=ir_utils.BEIRAdapter.get_corpus(df, "repo", text_cols),
        qrels=ir_utils.BEIRAdapter.get_qrels(df, "repo", "true_tasks"),
    )


def get_ir_results(ir_data):
    hostname = "localhost"  # localhost
    index_name = "github_search"  # scifact

    #### Intialize ####
    # (1) True - Delete existing index and re-index all documents from scratch
    # (2) False - Load existing index
    initialize = True  # False

    #### Sharding ####
    # (1) For datasets with small corpus (datasets ~ < 5k docs) => limit shards = 1
    # SciFact is a relatively small dataset! (limit shards to 1)
    number_of_shards = 1
    model = BM25(
        index_name=index_name,
        hostname=hostname,
        initialize=initialize,
        number_of_shards=number_of_shards,
    )

    # (2) For datasets with big corpus ==> keep default configuration
    # model = BM25(index_name=index_name, hostname=hostname, initialize=initialize)
    retriever = EvaluateRetrieval(model)

    # Retrieve dense results (format of results is identical to qrels)
    results = retriever.retrieve(ir_data.corpus, ir_data.queries)

    # Evaluate your retrieval using NDCG@k, MAP@K ...
    logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))

    ndcg, _map, recall, precision = retriever.evaluate(
        ir_data.qrels, results, retriever.k_values
    )
    return RetrievalResults(
        results=results, metrics={**ndcg, **_map, **recall, **precision}
    )


def print_ir_results_sample(ir_data, results):
    query_id, scores_dict = random.choice(list(ir_data.results.items()))
    logging.info("Query : %s\n" % ir_data.queries[query_id])

    scores = sorted(scores_dict.items(), key=lambda item: item[1], reverse=True)
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


class TextColConfig(BaseModel):
    text_cols: list = Field(default=[["dependencies", "tasks"]])


def main(text_cols_str=None, text_col_config_path: str = None):
    if text_col_config_path is None:
        if text_cols_str is not None:
            text_cols = [[s.strip() for s in text_cols_str.split(",")]]
        else:
            text_cols = [["dependencies", "tasks"]]
    else:
        with open(text_col_config_path) as f:
            text_col_config = TextColConfig.parse_obj(json.load(f))
            text_cols = text_col_config.text_cols

    metrics_records = []
    for cols in text_cols:
        ir_data = load_ir_data(generation_metrics_df, cols)
        ir_results = get_ir_results(ir_data)

        results = ir_results.results
        metrics = ir_results.metrics
        metrics["text_columns"] = cols
        metrics_records.append(metrics)
    #### Retrieval Example ####
    pd.DataFrame.from_records(metrics_records).to_json("output/beir_metrics.json")


if __name__ == "__main__":
    fire.Fire(main)
