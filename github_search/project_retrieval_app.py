import gradio as gr
import numpy as np
import pandas as pd

import attr
import pickle

from sklearn import metrics

from github_search.matching_zsl import *
from github_search import github_readmes
from github_search.retrieval_results import Retriever
import streamlit as st


def truncate_description(description, length=50):
    return " ".join(description.split()[:length])


def get_repos_with_descriptions(repos_df, repos):
    return repos_df.loc[repos]


def search_f(retriever: Retriever, query: str, k: int, description_length: int):
    results = retriever.retrieve_query_results(query, k)
    # results['repo'] = results.index
    results["link"] = "https://github.com/" + results.index
    results["description"] = results["description"].apply(
        lambda desc: truncate_description(desc, description_length)
    )
    return results.reset_index()


def show_retrieval_results(retriever: Retriever, query: str, k: int, description_length: int):
    print("started retrieval")
    if query in readme_data_test.y.values:
        with st.expander(
            "query is in gold standard set queries. Toggle viewing gold standard results?"
        ):
            st.write("gold standard results")
            task_repos = readme_data_test.repos[readme_data_test.y == query]
            st.table(get_repos_with_descriptions(retriever.X_df, task_repos))
    with st.spinner(text="fetching results"):
        st.table(search_f(retriever, query, k, description_length))
    print("finished retrieval")


if __name__ == "__main__":

    print("loading data")
    readme_data_test = pickle.load(open("output/readme_data_test.pkl", "rb"))
    readme_learner = pickle.load(open("output/readme_learner.pkl", "rb"))

    print("setting up retriever")
    readme_retriever = Retriever.from_retriever_learner(readme_learner)
    readme_retriever.set_embeddings(
        readme_data_test.repos,
        readme_data_test.X,
        readme_data_test.X,
        readme_data_test.all_tasks,
    )

    retrieved_results = st.sidebar.number_input("number of results", value=25)
    description_length = st.sidebar.number_input(
        "number of used description words", value=50
    )

    tasks_deduped = (
        readme_data_test.y.value_counts().reset_index()
    )  # drop_duplicates().sort_values().reset_index(drop=True)
    tasks_deduped.columns = ["task", "n_projects"]
    with st.sidebar.expander("Toggle viewing set queries"):
        st.table(tasks_deduped)

    query = st.text_input("input query", value="metric learning")
    show_retrieval_results(
        readme_retriever, query, retrieved_results, description_length
    )
