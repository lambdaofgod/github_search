import pickle

import attr
import numpy as np
import pandas as pd
from sklearn import metrics

import streamlit as st
from github_search import github_readmes
from findkit import retrieval_pipeline, feature_extractor
from findkit import retrieval_pipeline, index
from functools import partial
import sentence_transformers
import fire
from typing import List, Dict


def truncate_description(description, length=50):
    return " ".join(description.split()[:length])


def get_repos_with_descriptions(repos_df, repos):
    return repos_df.loc[repos]


def search_f(
    retrieval_pipe: retrieval_pipeline.RetrievalPipeline,
    query: str,
    k: int,
    description_length: int,
    doc_col: List[str],
):
    results = retrieval_pipe.find_similar(query, k)
    # results['repo'] = results.index
    results["link"] = "https://github.com/" + results["repo"]
    for col in doc_col:
        results[col] = results[col].apply(
            lambda desc: truncate_description(desc, description_length)
        )
    shown_cols = ["repo", "tasks", "link", "distance"]
    shown_cols = shown_cols + doc_col
    return results.reset_index(drop=True)[shown_cols]


def show_retrieval_results(
    retrieval_pipe: retrieval_pipeline.RetrievalPipeline,
    query: str,
    k: int,
    all_queries: List[str],
    description_length: int,
    repos_by_query: Dict[str, pd.DataFrame],
    doc_col: str,
):
    print("started retrieval")
    if query in all_queries:
        with st.expander(
            "query is in gold standard set queries. Toggle viewing gold standard results?"
        ):
            st.write("gold standard results")
            task_repos = repos_by_query.get_group(query)
            st.table(get_repos_with_descriptions(retrieval_pipe.X_df, task_repos))
    with st.spinner(text="fetching results"):
        st.write(
            search_f(retrieval_pipe, query, k, description_length, doc_col).to_html(
                escape=False, index=False
            ),
            unsafe_allow_html=True,
        )
    print("finished retrieval")


def setup_pipeline(
    extractor: feature_extractor.SentenceEncoderFeatureExtractor,
    documents_df: pd.DataFrame,
    text_col: str,
):
    retrieval_pipeline.RetrievalPipelineFactory.build(
        documents_df[text_col], metadata=documents_df
    )


@st.cache
def setup_retrieval_pipeline(
    query_encoder_path, document_encoder_path, documents, metadata
):
    document_encoder = feature_extractor.SentenceEncoderFeatureExtractor(
        sentence_transformers.SentenceTransformer(document_encoder_path)
    )
    query_encoder = feature_extractor.SentenceEncoderFeatureExtractor(
        sentence_transformers.SentenceTransformer(query_encoder_path)
    )
    retrieval_pipe = retrieval_pipeline.RetrievalPipelineFactory(
        feature_extractor=document_encoder,
        query_feature_extractor=query_encoder,
        index_factory=partial(index.NMSLIBIndex.build, distance="cosinesimil"),
    )
    return retrieval_pipe.build(documents, metadata=metadata)


def main(query_encoder_path, document_encoder_path, data_path):
    print("loading data")
    test_df = pd.read_csv(data_path)

    print("setting up retrieval_pipe")
    doc_col = "dependencies"
    retrieval_pipeline = setup_retrieval_pipeline(
        query_encoder_path, document_encoder_path, test_df[doc_col], test_df
    )
    retrieved_results = st.sidebar.number_input("number of results", value=10)
    description_length = st.sidebar.number_input(
        "number of used description words", value=10
    )

    tasks_deduped = (
        test_df["tasks"].explode().value_counts().reset_index()
    )  # drop_duplicates().sort_values().reset_index(drop=True)
    tasks_deduped.columns = ["task", "n_projects"]
    with st.sidebar.expander("Toggle viewing set queries"):
        st.table(tasks_deduped)

    additional_shown_cols = st.sidebar.multiselect(
        label="additional cols", options=[doc_col], default=doc_col
    )

    repos_by_query = test_df.explode("tasks").groupby("tasks")
    query = st.text_input("input query", value="metric learning")
    show_retrieval_results(
        retrieval_pipeline,
        query,
        retrieved_results,
        tasks_deduped["task"].to_list(),
        description_length,
        repos_by_query,
        additional_shown_cols,
    )


if __name__ == "__main__":
    fire.Fire(main)
