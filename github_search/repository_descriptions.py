# AUTOGENERATED! DO NOT EDIT! File to edit: notebooks/CodeSearchNet_Repository_Descriptions.ipynb (unless otherwise specified).

__all__ = ['get_all_codesearch_df', 'get_html', 'get_short_description', 'get_pypi_package_description',
           'get_pypi_repo_description', 'load_pypi_repo_descriptions']

# Cell
import os
import requests
from io import StringIO
import sys
import time
import tqdm

import pypi_cli
from sklearn import feature_extraction, metrics
import numpy as np
import pandas as pd
import bs4

import mlutil.parallel

import haystack.document_store.memory
import haystack.document_store.elasticsearch
from haystack import document_store

import haystack.retriever.sparse
from haystack import retriever

# Cell


def get_all_codesearch_df(data_dir):
    return pd.concat([
        pd.read_json(os.path.join(data_dir, split), lines=True) for split in ['train.jsonl', 'valid.jsonl', 'test.jsonl']
    ])

# Cell


def get_html(url):
    return requests.get(url).text


def get_short_description(repo):
    url = 'http://www.github.com/{}'.format(repo)
    html = get_html(url)
    parsed_html = bs4.BeautifulSoup(html)
    return parsed_html.find('title').get_text()

# Cell


def get_pypi_package_description(package_name, part=2):
    temp_out = StringIO()
    sys.stdout = temp_out
    try:
        pypi_cli.info([package_name])

    except:
        pass
    stdout = sys.stdout.getvalue().split('\n')
    if len(stdout) > part:
        description = stdout[part]
    else:
        description = None
    sys.stdout = sys.__stdout__
    return description

# Cell


def get_pypi_repo_description(repo):
    print(repo.split('/'))
    return get_pypi_package_description(repo.split('/')[1])

# Cell

def load_pypi_repo_descriptions(repos_descriptions_path='data/repo_pypi_descriptions.csv'):
    if not os.path.exists(repos_descriptions_path):
        t_start = time.time()
        pypi_descriptions_p = list(mlutil.parallel.mapp(get_pypi_repo_description, repos))
        t_end = time.time()

        #repos_with_descriptions = [repo for (repo, n) in zip(repos, pypi_descriptions_p) if not n is None]
        repos_descriptions = [(repo, desc) for (repo, desc) in zip(repos, pypi_descriptions_p) if not (desc is None or desc == '')]
        repos, descriptions = zip(*repos_descriptions)
        repos_descriptions_df = pd.DataFrame({
            'repo': repos,
            'pypi_description': descriptions
        })
        repos_descriptions_df.to_csv(repos_descriptions_path)
        print('loaded descriptions in', round((t_end - t_start) / 60, 2), 'minutes')
    else:
        repos_descriptions_df = pd.read_csv(repos_descriptions_path, index_col=0)
    return repos_descriptions_df