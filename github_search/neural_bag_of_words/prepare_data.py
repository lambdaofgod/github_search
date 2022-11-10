import numpy as np
import pandas as pd
from github_search import python_tokens, utils


def tokenize_python_code(code_text):
    """tokenize each word in code_text as python token"""
    toks = code_text.split()
    return [
        tok
        for raw_tok in code_text.split()
        for tok in python_tokens.tokenize_python(raw_tok)
    ]


def get_dependency_texts(dependency_records_df):
    dependency_records_df = dependency_records_df[
        dependency_records_df["source"] != "<ROOT>"
    ]
    return dependency_records_df.groupby("repo")["destination"].agg(" ".join)


def prepare_dependency_data(upstream, product):
    dependency_records_path = str(upstream["dependency_graph.prepare_records"])
    dependency_records_df = pd.read_feather(dependency_records_path)
    dep_texts = get_dependency_texts(dependency_records_df)
    pd.DataFrame(dep_texts).reset_index().to_feather(product["text"])
    pd.DataFrame(dep_texts).reset_index(drop=True).to_csv(product["raw_text"], index=False, header=False)

