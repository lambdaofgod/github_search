



class ExperimentParams:
    sampled_repos_per_task = 20
    min_repos_per_task = 10


def get_repos_for_query(query, repos_df):
    return repos_df[repos_df["tasks"].apply(lambda ts: query in ts)]


def get_queries(repos_df, min_query_count):
    all_queries = repos_df["query_tasks"].explode()
    qcounts = all_queries.value_counts()
    return qcounts[qcounts >= min_query_count].index.to_list()


def prepare_query_data(repos_df, min_query_count=5):
    task_queries = {
        str(i): query
        for (i, query) in enumerate(
            get_queries(repos_df, min_query_count=min_query_count)
        )
    }

    task_qrels = {
        qid: {
            str(corpus_id): 1
            for corpus_id in get_repos_for_query(task_queries[qid], repos_df).index
        }
        for qid in task_queries.keys()
    }
    return task_queries, task_qrels


def create_corpora_df(
    sampled_repos_df,
    librarian_signatures_df,
    generated_readmes,
    repomap_generated_readmes,
    flat_generated_readmes,
    flat_repomap_generated_readmes,
    selected_python_code_df,
    dependency_signature_generated_readmes=None,
    flat_dependency_signature_generated_readmes=None,
):

    # Rename flat dataframes to ensure proper column naming
    flat_generated_readmes_renamed = flat_generated_readmes.rename(
        columns={
            col: f"{col}_flat"
            for col in flat_generated_readmes.columns
            if col not in ["repo_name"]
        }
    )

    flat_repomap_generated_readmes_renamed = flat_repomap_generated_readmes.rename(
        columns={
            col: f"{col}_flat_repomap"
            for col in flat_repomap_generated_readmes.columns
            if col not in ["repo_name"]
        }
    )

    # Start with base dataframes
    corpora_df = sampled_repos_df[["repo", "readme", "repomap", "tasks", "query_tasks"]].merge(
        librarian_signatures_df[
            [
                "repo",
                "dependency_signature",
                "generated_tasks",
                "repository_signature",
            ]
        ],
        on="repo",
    )

    # Merge generated readmes and drop duplicate repo_name columns
    corpora_df = corpora_df.merge(
        generated_readmes, left_on="repo", right_on="repo_name"
    )
    corpora_df = corpora_df.drop(columns=["repo_name"])

    corpora_df = corpora_df.merge(
        repomap_generated_readmes,
        left_on="repo",
        right_on="repo_name",
        suffixes=("", "_repomap"),
    )
    corpora_df = corpora_df.drop(columns=["repo_name"])

    corpora_df = corpora_df.merge(
        flat_generated_readmes_renamed,
        left_on="repo",
        right_on="repo_name",
    )
    corpora_df = corpora_df.drop(columns=["repo_name"])

    corpora_df = corpora_df.merge(
        flat_repomap_generated_readmes_renamed,
        left_on="repo",
        right_on="repo_name",
    )
    corpora_df = corpora_df.drop(columns=["repo_name"])

    corpora_df = corpora_df.merge(
        selected_python_code_df, left_on="repo", right_on="repo_name"
    )
    corpora_df = corpora_df.drop(columns=["repo_name"])
    
    # Merge dependency signature generated readmes if provided
    if dependency_signature_generated_readmes is not None:
        corpora_df = corpora_df.merge(
            dependency_signature_generated_readmes,
            left_on="repo",
            right_on="repo_name",
            suffixes=("", "_dep_sig"),
        )
        corpora_df = corpora_df.drop(columns=["repo_name"])
    
    if flat_dependency_signature_generated_readmes is not None:
        flat_dep_sig_renamed = flat_dependency_signature_generated_readmes.rename(
            columns={
                col: f"{col}_flat_dep_sig"
                for col in flat_dependency_signature_generated_readmes.columns
                if col not in ["repo_name"]
            }
        )
        corpora_df = corpora_df.merge(
            flat_dep_sig_renamed,
            left_on="repo",
            right_on="repo_name",
        )
        corpora_df = corpora_df.drop(columns=["repo_name"])
    
    return corpora_df


def create_corpus_records(df, corpus_colnames, corpus_name, name_col="repo"):
    return (
        corpus_name,
        {
            str(i): {"text": "\n".join(row[corpus_colnames]), "title": row[name_col]}
            for (i, row) in df.iterrows()
        },
    )


def create_corpora(
    sampled_repos_df,
    librarian_signatures_df,
    generated_readmes,
    repomap_generated_readmes,
    flat_generated_readmes,
    flat_repomap_generated_readmes,
    selected_python_code_df,
    dependency_signature_generated_readmes=None,
    flat_dependency_signature_generated_readmes=None,
):
    corpora_df = create_corpora_df(
        sampled_repos_df,
        librarian_signatures_df,
        generated_readmes,
        repomap_generated_readmes,
        flat_generated_readmes,
        flat_repomap_generated_readmes,
        selected_python_code_df,
        dependency_signature_generated_readmes,
        flat_dependency_signature_generated_readmes,
    )
    basic_corpora = [
        create_corpus_records(corpora_df, [colname], colname)
        for colname in ["readme", "repomap", "selected_code"]
    ]
    librarian_corpora = [
        create_corpus_records(corpora_df, [colname], colname)
        for colname in [
            "dependency_signature",
            "repository_signature",
            "generated_tasks",
        ]
    ]
    readme_corpora_mappings = [
        ("answer", "code2doc_generated_readme"),
        ("context_history", "code2doc_files_summary"),
        ("answer_repomap", "repomap_code2doc_generated_readme"),
        ("context_history_repomap", "repomap_code2doc_files_summary"),
        ("answer_flat", "flat_code2doc_generated_readme"),
        # ("reasoning_flat", "flat_code2doc_files_summary"),
        ("answer_flat_repomap", "flat_repomap_code2doc_generated_readme"),
        # ("reasoning_flat_repomap", "flat_repomap_code2doc_files_summary"),
    ]
    
    # Add dependency signature mappings if data is provided
    if dependency_signature_generated_readmes is not None:
        readme_corpora_mappings.append(("answer_dep_sig", "dep_sig_code2doc_generated_readme"))
        readme_corpora_mappings.append(("context_history_dep_sig", "dep_sig_code2doc_files_summary"))
    
    if flat_dependency_signature_generated_readmes is not None:
        readme_corpora_mappings.append(("answer_flat_dep_sig", "flat_dep_sig_code2doc_generated_readme"))
    readme_corpora = [
        create_corpus_records(corpora_df, [colname], corpus_name)
        for (colname, corpus_name) in readme_corpora_mappings
    ]
    return dict(basic_corpora + librarian_corpora + readme_corpora)
