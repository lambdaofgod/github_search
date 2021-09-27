__all__ = ['get_modules_string', 'get_module_corpus', 'prepare_module_corpus', 'store_word_vectors']


import json
import pandas as pd
import tqdm
import numpy as np
import gensim
import ast

from github_search import parsing_imports, repository_descriptions, paperswithcode_tasks, python_call_graph
from sklearn import feature_extraction, decomposition


def get_modules_string(modules):
    public_modules = [mod for mod in modules if not mod[0] == '_']
    return ' '.join(public_modules)


def get_module_corpus(files_df):
    module_lists = []
    repos = []

    for __, row in tqdm.tqdm(files_df.iterrows(), total=len(files_df)):
        try:
            maybe_imports = parsing_imports.get_modules(row['content'])
            module_lists.append(list(set(maybe_imports)))
            repos.append(row['repo_name'])
        except SyntaxError as e:
            print(row['repo_name'], e)
    df = pd.DataFrame({'repo': repos, 'imports': module_lists})
    return df


def get_paperswithcode_with_imports_df(papers_with_repo_df, per_repo_imports):
    return papers_with_repo_df.merge(
        per_repo_imports,
        left_on='repo',
        right_index=True
    ).drop_duplicates('repo')


def prepare_paperswithcode_with_imports_df(product, upstream, python_file_paths):
    """
    add import data to python files csv
    """
    print("PYTHON FILE PATHS", python_file_paths)
    python_files_df = pd.concat([
        pd.read_csv(path)
        for path in python_file_paths
    ])
    print(python_files_df.shape)
    repo_names = python_files_df['repo_name']
    paperswithcode_df, all_papers_df = paperswithcode_tasks.get_paperswithcode_dfs()
    papers_with_repo_df = paperswithcode_tasks.get_papers_with_repo_df(all_papers_df, paperswithcode_df, repo_names)
    papers_with_repo_df = paperswithcode_tasks.get_papers_with_biggest_tasks(papers_with_repo_df, 500)
    import_corpus_df = pd.read_csv(upstream['prepare_module_corpus'])
    import_corpus_df['imports'] = import_corpus_df['imports'].apply(ast.literal_eval)
    per_repo_imports = import_corpus_df.groupby('repo')['imports'].agg(sum).apply(set)
    paperswithcode_with_imports_df = get_paperswithcode_with_imports_df(papers_with_repo_df, per_repo_imports)
    paperswithcode_with_imports_df.to_csv(str(product))


def prepare_module_corpus(python_file_paths, product):
    """
    prepare csv file with modules for import2vec
    """
    python_files_df = pd.concat([
        pd.read_csv(path)
        for path in python_file_paths
    ]).dropna(subset=['repo_name', 'content'])
    get_module_corpus(python_files_df).to_csv(str(product))


def prepare_dependency_records(python_file_paths, product):
    """
    prepare python dependency graph records (function calls, files in repo) csv
    """
    python_files_df = pd.concat([
        pd.read_csv(path)
        for path in python_file_paths
    ]).dropna(subset=['repo_name', 'content'])
    repo_dependency_graph_fetcher = python_call_graph.RepoDependencyGraphFetcher()
    dependency_records_df = repo_dependency_graph_fetcher.get_dependency_df(python_files_df)
    dependency_records_df['destination'] = np.where(dependency_records_df['edge_type'] == 'repo-file', dependency_records_df['destination'] + ":file", dependency_records_df['destination'])
    dependency_records_df['source'] = np.where(dependency_records_df['edge_type'] == 'file-function', dependency_records_df['source'] + ":file", dependency_records_df['source'])
    dependency_records_df.to_csv(str(product), index=False)


def _word_vectors_to_word2vec_format_generator(vocabulary, word_vectors):
    for (word, vector) in zip(vocabulary, word_vectors):
        yield word + ' ' + ' '.join([str('{:.5f}'.format(f)) for f in vector])


def store_word_vectors(words, word_vectors, file_name):
    with open(file_name, 'w') as f:
        f.write(str(len(words)) + ' ' + str(word_vectors.shape[1]) + '\n')
        for line in _word_vectors_to_word2vec_format_generator(words, module_vectors):
            f.write(line + '\n')
