import os

import pandas as pd
from haystack.nodes import retriever as haystack_retriever
from sentence_transformers import InputExample, SentenceTransformer, evaluation, models
from toolz import partial

from github_search.ir import evaluator
from github_search import utils

def make_search_df(df, ir_config, doc_col):
    return utils.concatenate_flattened_list_cols(
        df,
        [ir_config.query_col] + ir_config.document_cols,
        ir_config.document_cols,
        doc_col,
    )


def get_ir_metrics(path):
    metrics_df = pd.read_csv(
        os.path.join(path, "Information-Retrieval_evaluation_results.csv")
    )
    return metrics_df[[col for col in metrics_df if "cos" in col]]


def get_result_metadata(retriever: haystack_retriever.BaseRetriever, query, topk=100):
    return [
        {**doc.meta, "score": doc.score}
        for doc in retriever.retrieve(query, top_k=topk)
    ]


def get_normalized_ngram_distance(reference, other):
    reference_tokens = reference.split()
    return 1 - len(set(reference_tokens).intersection(set(other.split()))) / len(
        reference_tokens
    )


def get_tasks_at_ngram_overlap(tasks, reference_task, threshold=0.5):
    return [
        task
        for task in tasks
        if get_normalized_ngram_distance(reference_task, task) <= threshold
    ]


def get_aggregated_results_df(
    retriever: haystack_retriever.BaseRetriever, query, topk, prefilter_topk=100
):
    query_results_df = pd.DataFrame(
        get_result_metadata(retriever, query, topk=prefilter_topk)
    )
    if len(query_results_df) == 0:
        return
    else:
        return (
            query_results_df.groupby("repo")
            .agg({"score": "mean", "tasks": "first"})
            .sort_values("score")
            .iloc[:topk]
        )


def get_accuracy(matched_tasks):
    return any([tasks for tasks in matched_tasks if len(tasks) > 0])


def get_precision(matched_tasks, topk):
    return len([tasks for tasks in matched_tasks if len(tasks) > 0]) / topk


def evaluate_query_results_df(query, results_df, thresholds, topk=10):
    """
    evaluates query results at thresholds
    """
    matched_tasks = results_df["tasks"].apply(
        partial(get_tasks_at_ngram_overlap, reference_task=query, threshold=0.0)
    )
    partially_matched_tasks = {
        threshold: results_df["tasks"].apply(
            partial(
                get_tasks_at_ngram_overlap, reference_task=query, threshold=threshold
            )
        )
        for threshold in thresholds
    }
    return {
        "accuracy": 1.0 * get_accuracy(matched_tasks),
        **{
            f"accuracy@overlap={threshold}": 1.0
            * get_accuracy(partially_matched_tasks[threshold])
            for threshold in partially_matched_tasks.keys()
        },
        "precision": get_precision(matched_tasks, topk),
        **{
            f"precision@overlap={threshold}": get_precision(
                partially_matched_tasks[threshold], topk
            )
            for threshold in partially_matched_tasks.keys()
        },
    }


def evaluate_query_results(
    retriever: haystack_retriever.BaseRetriever, query, thresholds, topk=10
):
    df = get_aggregated_results_df(retriever, query, topk)
    if df is None:
        return {
            "accuracy": 0,
            **{f"accuracy@overlap={threshold}": 0 for threshold in thresholds},
            "precision": 0,
            **{f"precision@overlap={threshold}": 0 for threshold in thresholds},
        }
    return evaluate_query_results_df(query, df, thresholds)
