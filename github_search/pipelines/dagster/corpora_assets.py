from pathlib import Path

import pandas as pd
from beir.retrieval.search.lexical import BM25Search as BM25
from github_search.evaluation.beir_evaluation import (
    EvaluateRetrievalCustom as CorpusDataLoader,
)
from github_search.evaluation.corpus_utils import *

data_path = Path("../output").expanduser()

librarian_signatures_df = pd.read_parquet(
    "/home/kuba/Projects/uhackathons/fastrag_util/data/librarian_signatures.parquet"
)

model_name = "codellama_repomaps"
sample_prefix = "sample_per_task_5_repos"


bigger_sample_path = f"../output/code2doc/sample_per_task_5_repos/sampled_repos{ExperimentParams.sampled_repos_per_task}.jsonl"
sample_path = (
    bigger_sample_path  # "../output/code2doc/sample_small/sampled_repos_min10.jsonl"
)
sampled_repos_df = pd.read_json(sample_path, orient="records", lines=True)
sample_python_code_df = pd.read_feather(
    Path(data_path) / "code" / "python_files_with_selected_code.feather"
)


repos_with_all_data = (
    set(sampled_repos_df["repo"])
    & set(librarian_signatures_df["repo"])
    & set(sample_python_code_df["repo_name"])
)


len(repos_with_all_data)


# librarian_signatures_df = librarian_signatures_df[librarian_signatures_df["generation"] == 0]


# Select only repos with signatures that were in sample


sampled_repos_df, sample_python_code_df, sampled_librarian_signatures_df = (
    filter_dfs_by_cols_in(
        [sampled_repos_df, sample_python_code_df, librarian_signatures_df],
        repos_with_all_data,
    )
)
sampled_repos_df, sampled_librarian_signatures_df = align_dfs(
    [sampled_repos_df, sampled_librarian_signatures_df]
)


# ## Sample with generated READMEs

sample_loader = CorpusDataLoader(
    repos_df_path=data_path / f"code2doc/{sample_prefix}/sampled_repos5.jsonl",
    generated_readmes_df_path=data_path
    / f"code2doc/{sample_prefix}/{model_name}_generated_readmes5.jsonl",
    code_df_path=data_path / "code" / "python_files_with_selected_code.feather",
)


sampled_repos_df, sampled_generated_readmes_df, sample_python_code_df = (
    sample_loader.load_corpus_dfs(librarian_signatures_df["repo"])
)


repos_with_all_data = set(sampled_repos_df["repo"]).intersection(
    librarian_signatures_df["repo"]
)


sampled_repos_df, sample_python_code_df, sampled_librarian_signatures_df = (
    filter_dfs_by_cols_in(
        [sampled_repos_df, sample_python_code_df, librarian_signatures_df],
        repos_with_all_data,
    )
)
sampled_repos_df, sampled_librarian_signatures_df = align_dfs(
    [sampled_repos_df, sampled_librarian_signatures_df]
)  # [sampled_librarian_signatures_df["generation"] == 0]])


task_queries, task_qrels = prepare_query_data(
    sampled_repos_df, min_query_count=ExperimentParams.min_repos_per_task
)


pd.Series(task_qrels).apply(len).describe()


pd.Series([len(qrl) for qrl in task_qrels.values()]).describe()


# corpora = prepare_basic_corpora(sampled_repos_df, sample_python_code_df) |  #
corpora = prepare_corpora(
    sampled_repos_df, sampled_generated_readmes_df, sample_python_code_df
) | prepare_librarian_corpora(sampled_repos_df, sampled_librarian_signatures_df)
