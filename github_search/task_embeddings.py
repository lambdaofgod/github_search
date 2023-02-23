#!/usr/bin/env python
# coding: utf-8

import logging
import os
import pickle

import fasttext
import gensim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import sentence_transformers
import umap
from mlutil import feature_extraction_utils
from mlutil.feature_extraction import embeddings
from sentence_transformers import models as sbert_models
from sklearn.metrics import pairwise

import streamlit as st
from github_search import paperswithcode_tasks

pio.renderers.default = "iframe"


logging.basicConfig(level="info")

plt.style.use("dark_background")


# os.chdir("../")


area_grouped_tasks = pd.read_csv("data/paperswithcode_tasks.csv").dropna()


area_grouped_tasks["task"] = area_grouped_tasks["task"].apply(
    paperswithcode_tasks.clean_task_name
)


lstm_model_path = "output/sbert/lstm2x256_epoch325/"
codebert_model_path = "output/sbert/codebert15/"


def make_2d_data_plot(data, text_label, cls):
    cls_numbering = {c: i for (i, c) in enumerate(set(cls))}
    source_df = pd.DataFrame(
        {
            "x": data[:, 0],
            "y": data[:, 1],
            "task": text_label,
            "area": cls,
            "color": [cls_numbering[c] for c in cls],
        }
    )
    plot = px.scatter(
        data_frame=source_df,
        x="x",
        y="y",
        hover_data=["task", "area"],
        color="area",
        width=1200,
        height=800,
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )

    return plot


reduced_features = pickle.load(open("output/reduced_features.pkl", "rb"))

logging.info("making w2v figure")
w2v_fig = make_2d_data_plot(
    reduced_features["word2vec"],
    area_grouped_tasks["task"],
    area_grouped_tasks["area"],
)
st.write("# Word2Vec")
st.write(w2v_fig)


logging.info("making fasttext figure")
fasttext_fig = make_2d_data_plot(
    reduced_features["fasttext"],
    area_grouped_tasks["task"],
    area_grouped_tasks["area"],
)
st.write("# Fasttext")
st.write(fasttext_fig)


logging.info("making lstm figure")
lstm_model_path = "output/sbert/lstm2x256_epoch325/"


lstm_ir_metrics = pd.read_csv(
    os.path.join(lstm_model_path, "Information-Retrieval_evaluation_results.csv")
)
lstm_fig = make_2d_data_plot(
    reduced_features["lstm"],
    area_grouped_tasks["task"],
    area_grouped_tasks["area"],
)
st.write("# LSTM")
st.write(lstm_ir_metrics[[col for col in lstm_ir_metrics if "cos_sim" in col]].T)
st.write(lstm_fig)

codebert_ir_metrics = pd.read_csv(
    os.path.join(codebert_model_path, "Information-Retrieval_evaluation_results.csv")
)
logging.info("making codebert figure")


codebert_fig = make_2d_data_plot(
    reduced_features["codebert"],
    area_grouped_tasks["task"],
    area_grouped_tasks["area"],
)

st.write("# CodeBert")
st.write(codebert_ir_metrics[[col for col in lstm_ir_metrics if "cos_sim" in col]].T)
st.write(codebert_fig)
