import re
import shutil
from pathlib import Path as P
import sentence_transformers
import yaml
from findkit import index, feature_extractor
from github_search.ir import ir_utils, evaluator
from typing import List, Dict, Set
import tqdm
import ast

from github_search import utils
from github_search.papers_with_code import paperswithcode_tasks
from github_search.neural_bag_of_words import embedders
from github_search.neural_bag_of_words import evaluation
import sentence_transformers
from github_search.neural_bag_of_words.evaluation import *


## prepare_data


df_dep_texts = pd.read_parquet("../../output/nbow_data_test.parquet")
df_dep_texts["tasks"] = df_dep_texts["tasks"].apply(ast.literal_eval)


# ## Combining evaluation


upstream_paths = list(P("../../output/models/").glob("nbow*"))


upstream_paths


upstream_path = P("../../output/models/nbow_mean-1")


def prepare_display_df(metrics_df):
    return metrics_df.sort_values("accuracy@10", ascending=False)


def get_eval_df(upstream_paths):
    return evaluation.get_metrics_df_from_dicts(
        dict(
            evaluation.get_metrics_dict_with_name(
                p / "metrics_final.yaml", get_name=lambda arg: p
            )
            for p in upstream_paths
        )
    )


eval_df = prepare_display_df(get_eval_df(upstream_paths))


eval_df.style.highlight_max()


def get_nbow_eval_results(nbow_paths):
    dfs = [pd.read_csv(p / "metrics_final.yaml") for p in nbow_paths]
    for p, df in zip(nbow_paths, dfs):
        df["path"] = p
    return pd.concat(dfs).set_index("path")


best_model_directory = eval_df.index[0]


def get_best_model_paths(best_model_directory):
    return {
        "query_nbow": best_model_directory / "nbow_query_final",
        "document_nbow": best_model_directory / "nbow_document_final",
    }


best_model_paths = get_best_model_paths(eval_df.index[0])


query_embedder = sentence_transformers.SentenceTransformer(
    best_model_paths["query_nbow"]
)
document_embedder = sentence_transformers.SentenceTransformer(
    best_model_paths["document_nbow"]
)


document_embedder.max_seq_length = 5000
query_embedder.max_seq_length = 100


searched_dep_texts = df_dep_texts


ir_evaluator = evaluator.InformationRetrievalEvaluator(
    query_embedder, document_embedder
)


ir_evaluator.setup(searched_dep_texts, "tasks", "dependencies")


ir_metrics = ir_evaluator.evaluate()


print(yaml.dump(ir_metrics["cos_sim"]))


tasks = searched_dep_texts["tasks"].explode().drop_duplicates()


ir_evaluator.get_ir_results


queries, corpus, relevant_docs = evaluator.get_ir_dicts(
    searched_dep_texts, "tasks", "dependencies"
).values()
ir_evaluator = evaluator.get_ir_evaluator(searched_dep_texts, doc_col="dependencies")


result_lists = ir_evaluator.get_queries_result_lists(query_embedder, document_embedder)


def get_retrieval_prediction_dfs(
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


def get_golden_result_lists(relevant_docs: Dict[str, Set[str]]):
    golden_result_lists = [
        [{"corpus_id": corpus_id, "score": 1} for corpus_id in corpus_ids]
        for (q_id, corpus_ids) in relevant_docs.items()
    ]
    return golden_result_lists


def get_per_query_hits_at_k(predicted_documents, relevant_docs, k=10):
    return {
        q: int(
            k
            * predicted_results[q]
            .sort_values("score", ascending=False)
            .iloc[:k]["corpus_id"]
            .isin(relevant_docs[q_id])
            .mean()
        )
        for (q_id, q) in queries.items()
    }


# # import itertools
# list(itertools.islice(relevant_docs.items(),1))


from collections import Counter


import yaml
import json
import pprint


def round_float_dict(d, rounding=3):
    if type(d) is dict:
        return {k: round_float_dict(v) for k, v in d.items()}
    else:
        return float(round(d, rounding))


predicted_results = get_retrieval_prediction_dfs(
    queries, searched_dep_texts["repo"].to_list(), result_lists["cos_sim"]
)


predicted_results["recommendation systems"]


hits_at_10 = get_per_query_hits_at_k(predicted_results, relevant_docs)


task_df = pd.read_csv("../../data/paperswithcode_tasks.csv")


raw_tasks_with_hits_df = (
    task_df.merge(pd.Series(hits_at_10, name="hits"), left_on="task", right_index=True)
    .drop(columns=["task_description"])
    .drop_duplicates()
)


def get_extremal_tasks_with_hits(
    raw_tasks_with_hits_df, get_best=True, n_tasks_per_area=10
):
    tasks_with_hits_df = raw_tasks_with_hits_df.groupby("area").apply(
        lambda df: df.sort_values("hits", ascending=not get_best).iloc[
            :n_tasks_per_area
        ]
    )
    tasks_with_hits_df.index = tasks_with_hits_df.index.get_level_values("area")
    tasks_with_hits_df = tasks_with_hits_df.drop(columns="area")
    return tasks_with_hits_df


best_tasks_with_hits_df = get_extremal_tasks_with_hits(raw_tasks_with_hits_df)


worst_tasks_with_hits_df = get_extremal_tasks_with_hits(
    raw_tasks_with_hits_df, get_best=False
)


best_tasks_with_hits_df.to_csv("../../output/best_tasks_with_hits.csv")


worst_tasks_with_hits_df.to_csv("../../output/worst_tasks_with_hits.csv")


searched_dep_texts["tasks"].explode().value_counts().describe()


query_vectors = query_embedder.encode(
    tasks.to_list(), show_progress_bar=True, convert_to_tensor=False
)


query_vectors.shape


tasks


document_embedder = document_embedder.cuda()


with torch.cuda.amp.autocast():
    dep_vectors = document_embedder.encode(
        searched_dep_texts["dependencies"].values,
        batch_size=64,
        convert_to_tensor=False,
        show_progress_bar=True,
    )


# searched_dep_texts["tasks"].explode().unique().shape


deps_index = index.NMSLIBIndex.build(
    dep_vectors, searched_dep_texts, distance="cosinesimil"
)


i = 7


print("query: " + tasks.iloc[i])
deps_index.find_similar(query_vectors[i], 10)


q = "model compression"


q_vec = query_embedder.encode(q)


# query_index = index.NMSLIBIndex.build(query_vectors, pd.DataFrame({"task": tasks}), distance="angulardist")


i = 4
print(q)
deps_index.find_similar(
    q_vec, 10
)  # .drop(columns=["dependencies"])#.drop_duplicates(subset="repo")
