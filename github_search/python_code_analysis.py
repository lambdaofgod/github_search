import ast
import logging
from collections import namedtuple
from functools import partial

import astunparse
import pandas as pd
import tqdm
from comment_parser import comment_parser
from sklearn import feature_extraction, metrics

from github_search import utils

Import = namedtuple("Import", ["module", "name", "alias"])
COMMENT_SEP = "#<NEWCOMMENT>"


def _get_imports(file_content):
    root = ast.parse(file_content)

    for node in ast.iter_child_nodes(root):
        if isinstance(node, ast.Import):
            module = []
        elif isinstance(node, ast.ImportFrom) and not node.module is None:
            module = node.module.split(".")
        else:
            continue

        for n in node.names:
            yield Import(module, n.name.split("."), n.asname)


def get_import_expressions(file_contents):
    root = ast.parse(file_contents)
    for node in ast.iter_child_nodes(root):
        if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
            yield node


def get_imports(file_contents):
    return "\n".join(
        [
            astunparse.unparse(imp).strip()
            for imp in get_import_expressions(file_contents)
        ]
    )


def get_module_from_import(imp):
    if imp.module == []:
        return imp.name[0]
    else:
        return imp.module[0]


def get_modules(file_content):
    for imp in _get_imports(file_content):
        yield get_module_from_import(imp)


def get_per_repo_similarities(
    paperswithcode_df: pd.DataFrame,
    repo_grouped_contents: dict,
    similar_col: str,
    bow_vectorizer_class: str,
):
    bow_vectorizer = getattr(feature_extraction.text, bow_vectorizer_class)()

    bow_vectorizer.fit(paperswithcode_df[similar_col])
    return {
        repo: pd.Series(
            data=metrics.pairwise.cosine_similarity(
                bow_vectorizer.transform([abstract]),
                bow_vectorizer.transform(repo_grouped_contents[repo]["content"]),
            )[0],
            index=repo_grouped_contents[repo]["path"],
        )
        for (repo, abstract) in tqdm.tqdm(
            paperswithcode_df[["repo", similar_col]].itertuples(index=False),
            total=paperswithcode_df.shape[0],
        )
        if repo_grouped_contents.get(repo) is not None
        and len(repo_grouped_contents[repo]) > 0
    }


def get_top_similar_file_paths_df(
    python_files_df: pd.DataFrame,
    paperswithcode_df: pd.DataFrame,
    similar_col: str,
    files_per_repo: int,
    bow_vectorizer_class: str,
):
    selected_python_files_df = python_files_df[
        python_files_df["repo_name"].isin(paperswithcode_df["repo"])
    ]
    repo_grouped_contents = dict(
        tuple(selected_python_files_df.groupby("repo_name")[["content", "path"]])
    )
    per_repo_file_similarities = get_per_repo_similarities(
        paperswithcode_df, repo_grouped_contents, similar_col, bow_vectorizer_class
    )
    top_similar_file_paths = {
        key: per_repo_file_similarities[key].sort_values(ascending=False)[
            :files_per_repo
        ]
        for key in per_repo_file_similarities.keys()
    }
    top_similar_files_dfs = [
        pd.DataFrame(
            {
                "repo": [repo] * len(top_similar_file_paths[repo]),
                "similarity": top_similar_file_paths[repo].values,
                "path": top_similar_file_paths[repo].index,
            }
        )
        for repo in top_similar_file_paths.keys()
    ]
    top_similar_files_dfs = [
        df.merge(
            repo_grouped_contents[df["repo"].iloc[0]][["content", "path"]], on="path"
        )
        for df in top_similar_files_dfs
    ]
    return pd.concat(top_similar_files_dfs).reset_index(drop=True)


def select_repo_files(
    python_files_path: str,
    upstream: dict,
    similar_col: str,
    files_per_repo: int,
    product: str,
    bow_vectorizer_class: str,
):
    logging.info("loading python files")
    paperswithcode_path = str(upstream["make_readmes"])
    python_files_df = pd.read_feather(python_files_path).dropna()
    logging.info("loading paperswithcode df")
    paperswithcode_with_tasks_df = pd.read_csv(paperswithcode_path).dropna(
        subset=["least_common_task", similar_col]
    )
    paperswithcode_with_tasks_df["tasks"] = paperswithcode_with_tasks_df["tasks"].apply(
        ast.literal_eval
    )
    logging.info("selecting files")
    top_similar_files = get_top_similar_file_paths_df(
        python_files_df,
        paperswithcode_with_tasks_df,
        similar_col=similar_col,
        files_per_repo=files_per_repo,
        bow_vectorizer_class=bow_vectorizer_class,
    )
    top_similar_files.drop_duplicates(["repo", "path"]).reset_index(
        drop=True
    ).to_feather(str(product))


def get_docstrings(file_contents):
    root = ast.parse(file_contents)
    for node in ast.iter_child_nodes(root):
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.ClassDef):
            maybe_docstring = ast.get_docstring(node)
            if maybe_docstring is not None:
                yield maybe_docstring


def segment_contiguous(items, indices, neighbor_threshold):
    if len(items) == 0:
        return []
    returned_items = []
    tmp = [items[0]]
    for i in range(1, len(items)):
        if indices[i] - indices[i - 1] <= neighbor_threshold:
            tmp.append(items[i])
        else:
            returned_items.append(tmp)
            tmp = [items[i]]
    returned_items.append(tmp)
    return returned_items


def get_comments_and_docstrings(file_contents, neighbor_threshold):
    docstrings = list(get_docstrings(file_contents))
    comments = comment_parser.extract_comments_from_str(
        file_contents, mime="text/x-script.python"
    )
    comment_texts = [com.text() for com in comments]
    comment_line_nos = [com.line_number() for com in comments]
    return docstrings + [
        "\n".join(seg).strip()
        for seg in segment_contiguous(
            comment_texts, comment_line_nos, neighbor_threshold
        )
    ]


def extract_python_comments(upstream, product, line_neighbor_threshold):
    files_df = pd.read_feather(str(upstream["select_repo_files"]))
    comments = [
        utils.try_run(
            partial(
                get_comments_and_docstrings, neighbor_threshold=line_neighbor_threshold
            ),
            default=[],
        )(file_contents)
        for file_contents in tqdm.tqdm(files_df["content"])
    ]
    files_df["comments"] = pd.Series(comments).apply(("\n" + COMMENT_SEP + "\n").join)
    files_df = files_df[["repo", "path", "comments"]]
    files_df.to_feather(str(product))


def extract_python_imports(upstream, product):
    files_df = pd.read_feather(str(upstream["select_repo_files"]))
    imports = [
        utils.try_run(
            partial(get_imports),
            default="",
        )(file_contents)
        for file_contents in tqdm.tqdm(files_df["content"])
    ]
    files_df["imports"] = pd.Series(imports)
    files_df = files_df[["repo", "path", "imports"]]
    print(set(files_df["imports"].apply(type)))
    files_df.to_feather(str(product))
