import os

import pandas as pd
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


class BEIRAdapter:
    @classmethod
    def get_corpus_df(cls, generation_metrics_df, id_col, text_cols):
        df = generation_metrics_df.copy()
        df = df.set_index(id_col)
        df["text"] = ""
        for col in text_cols:
            df["text"] = df["text"] + " " + df[col]
        return df

    @classmethod
    def get_corpus(cls, generation_metrics_df, id_col, text_cols):
        corpus_df = cls.get_corpus_df(generation_metrics_df, id_col, text_cols)
        rs = list(corpus_df.head().iterrows())
        return {
            id: {"text": row["text"], "true_tasks": row["true_tasks"]}
            for (id, row) in corpus_df.iterrows()
        }

    @classmethod
    def get_queries(cls, generation_metrics_df, query_col):
        queries_list = (
            generation_metrics_df[query_col].explode().drop_duplicates().tolist()
        )
        return {q: q for q in queries_list}

    @classmethod
    def _get_qrels_values(cls, doc_ids):
        return {doc_id: 1 for doc_id in doc_ids.tolist()}

    @classmethod
    def get_qrels(cls, df, id_col, query_col):
        query_gb = df[[id_col, query_col]].explode(query_col).groupby(query_col)
        return {
            name: cls._get_qrels_values(group[id_col]) for (name, group) in query_gb
        }
