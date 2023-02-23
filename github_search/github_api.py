import os

import requests
from pandas.io.json import json_normalize


def get_searched_repositories_df(query, token_file_path=".github_token", pages=10):
    repo_information = search_repositories(query, token_file_path, pages)
    return get_cleaned_repositories_df(repo_information)


def search_repositories(query, token_file_path=".github_token", pages=10):
    headers = _get_headers(token_file_path)
    i = 1
    url = "https://api.github.com/search/repositories?q={}&page={}&per_page=100"
    starred_response = []
    tmp_response = requests.get(
        url.format(query, i),
        headers={**headers, "Accept": "application/vnd.github.mercy-preview+json"},
    )
    while tmp_response.ok and i <= pages:
        starred_response = starred_response + tmp_response.json()["items"]
        i += 1
        tmp_response = requests.get(
            url.format(query, i),
            headers={**headers, "Accept": "application/vnd.github.mercy-preview+json"},
        )

    print(tmp_response.json())
    if len(starred_response) == 0:
        raise requests.HTTPError(
            "Error occured while fetching, most likely you went over rate limit"
        )
    else:
        return starred_response


def get_cleaned_repositories_df(repo_information):
    repo_df = json_normalize(repo_information)
    repo_df = repo_df.drop_duplicates(subset=["name"])
    repo_df.index = repo_df["name"]
    repo_df.drop("name", axis=1, inplace=True)
    repo_df["topics"] = repo_df["topics"].apply(" ".join)
    repo_df["description"] = repo_df["description"].fillna("")
    return repo_df


def _get_headers(token_file_path):
    if os.path.exists(token_file_path):
        token = open(token_file_path).read().rstrip()
        return {"Authorization": "token %s" % token}
    else:
        return {}
