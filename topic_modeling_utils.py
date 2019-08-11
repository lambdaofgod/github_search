import os

import re
import tqdm
import requests

import numpy as np

from bs4 import BeautifulSoup
from markdown import markdown


import pandas as pd
from pandas.io.json import json_normalize
from gensim import summarization

from sklearn import decomposition, feature_extraction, manifold, pipeline
from sklearn.feature_extraction import stop_words
from sklearn import pipeline


from mlutil import topic_modeling

import seaborn as sns
import wordcloud

import matplotlib.pyplot as plt
from IPython.display import Markdown, display

import scipy.stats


def search_repositories(query, pages=10):
    i = 1
    url = 'https://api.github.com/search/repositories?q={}&page={}&per_page=100'
    starred_response = []
    tmp_response = requests.get(url.format(query, i), headers={**headers, "Accept": "application/vnd.github.mercy-preview+json"})
    while tmp_response.ok and i <= pages:
        starred_response = starred_response + tmp_response.json()['items']
        i += 1
        tmp_response = requests.get(url.format(query, i), headers={**headers, "Accept": "application/vnd.github.mercy-preview+json"})

    print(tmp_response.json())
    if len(starred_response) == 0:
        raise requests.HTTPError('Error occured while fetching, most likely you went over rate limit')
    else:
        return starred_response


def get_cleaned_repositories_df(repo_information):
    repo_df = json_normalize(repo_information)
    repo_df = repo_df.drop_duplicates(subset=['name'])
    repo_df.index = repo_df['name']
    repo_df.drop('name', axis=1, inplace=True)
    repo_df['topics'] = repo_df['topics'].apply(' '.join)
    repo_df['description'] = repo_df['description'].fillna('')
    return repo_df


def get_word_cloud(texts):
    text = ' '.join(texts)
    return wordcloud.WordCloud(max_font_size=40).generate(text)


def show_word_cloud(wc, figure_kwargs={'figsize': (8, 5)}):
    plt.figure(**figure_kwargs)
    plt.imshow(wc)
    plt.axis('off')
    plt.show()


def show_word_cloud_from_texts(text_column):
    texts = text_column.fillna('').values
    cloud = get_word_cloud(texts)
    show_word_cloud(cloud)


def get_topic_representant_indices(topic_weights, topic_idx, num_representants=5):
    indices = topic_weights[:, topic_idx].argsort()[::-1]
    return indices[:num_representants]


def get_repos_representing_topic(repo_df, topic_weights, topic_idx, num_representants=5):
    return repo_df.iloc[get_topic_representant_indices(topic_weights, topic_idx, num_representants)]