import ast

import numpy as np
import pandas as pd
from sklearn import metrics
import itertools
from operator import itemgetter


def iunzip(iterable, n=2):
    n_iterable = list(itertools.tee(iterable, n))
    return [map(itemgetter(i), p) for i, p in enumerate(n_iterable)]


def load_paperswithcode_df(path, drop_na_cols=["readme", "tasks"]):
    df = pd.read_csv(path)
    drop_na_cols = [col for col in drop_na_cols if col in df.columns]
    df = df.dropna(subset=drop_na_cols)
    if "tasks" in df.columns:
        df["tasks"] = df["tasks"].apply(ast.literal_eval)
    return df


def add_col_for_repo_from_paperswithcode(
    df, paperswithcode_df, added_cols=["tasks"], repo_col="repo"
):
    return df.merge(paperswithcode_df, left_on=repo_col, right_on="repo")[
        df.columns.to_list() + added_cols
    ]


def maybe_convert_to_numpy(x, dtype=np.int32):
    if type(x) is list:
        return np.array(x, dtype)
    else:
        return x


def get_accuracy_from_scores(labels, scores):
    labels = maybe_convert_to_numpy(labels)
    return np.mean(labels == scores.argmax(axis=1))


def get_multilabel_samplewise_topk_accuracy(labels: np.ndarray, scores: np.ndarray):
    """
    labels: np.ndarray, one-hot encoded labels
    scores: np.ndarray

    returns multilabel topk accuracy: for each record, it has k classes
    and we check how many from top k classes by score match them
    """
    labels = maybe_convert_to_numpy(labels)

    assert labels.shape == scores.shape
    accs = []
    scores_indices = np.argsort(-scores, axis=1)
    labels_indices = np.argsort(-labels, axis=1)
    total_n_classes = (labels > 0).sum()
    for (labels_row, scores_indices_row, labels_indices_row) in zip(
        labels, scores_indices, labels_indices
    ):
        n_classes = (labels_row > 0).sum()
        top_scores_indices = scores_indices_row[:n_classes]
        correct_labels_indices = labels_indices_row[:n_classes]
        row_topk_hits = len(
            set(correct_labels_indices).intersection(top_scores_indices)
        )
        accs.append(row_topk_hits)
    return sum(accs) / total_n_classes


def try_run(f, default=None):
    def _maybe_failed_f(args):
        try:
            return f(args)
        except:
            return default

    return _maybe_failed_f


def get_current_memory_usage():
    """Memory usage in GB"""

    with open("/proc/self/status") as f:
        memusage = f.read().split("VmRSS:")[1].split("\n")[0][:-3]

    return round(int(memusage.strip()) / 1024 ** 2, 2)
