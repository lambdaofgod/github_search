#!/usr/bin/env python
# coding: utf-8
###############################################
# This script depends on
# step prepare_reduced_embeddings from pipeline
###############################################
import logging
import os
import pickle

import fasttext
import gensim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import sentence_transformers
import umap
from mlutil import feature_extraction_utils
from mlutil.feature_extraction import embeddings
from sentence_transformers import models as sbert_models
from sklearn.metrics import pairwise

import streamlit as st
from github_search import paperswithcode_tasks, visualization_utils

pio.renderers.default = "iframe"


logging.basicConfig(level="info")

plt.style.use("dark_background")


area_grouped_tasks = pd.read_csv("data/paperswithcode_tasks.csv").dropna()


area_grouped_tasks["task"] = area_grouped_tasks["task"].apply(
    paperswithcode_tasks.clean_task_name
)


lstm_model_path = "output/sbert/lstm2x256_epoch500/"
codebert_model_path = "output/sbert/codebert15/"


reduced_features = pickle.load(open("output/reduced_features.pkl", "rb"))

logging.info("making w2v figure")
w2v_fig = visualization_utils.make_2d_data_plot(
    reduced_features["word2vec"],
    area_grouped_tasks["task"],
    area_grouped_tasks["area"],
)
st.write("# Word2Vec")
st.write(w2v_fig)


logging.info("making fasttext figure")
fasttext_fig = visualization_utils.make_2d_data_plot(
    reduced_features["fasttext"],
    area_grouped_tasks["task"],
    area_grouped_tasks["area"],
)
st.write("# Fasttext")
st.write(fasttext_fig)


logging.info("making lstm figure")


lstm_ir_metrics_path = os.path.join(
    lstm_model_path, "Information-Retrieval_evaluation_results.csv"
)
lstm_ir_metrics = pd.read_csv(lstm_ir_metrics_path)
lstm_fig = visualization_utils.make_2d_data_plot(
    reduced_features["lstm"],
    area_grouped_tasks["task"],
    area_grouped_tasks["area"],
)
st.write("# LSTM")
st.write("information retrieval metrics loaded from {}".format(lstm_ir_metrics_path))
st.write(lstm_ir_metrics[[col for col in lstm_ir_metrics if "cos_sim" in col]].T)
st.write(lstm_fig)

codebert_ir_metrics_path = os.path.join(
    codebert_model_path, "Information-Retrieval_evaluation_results.csv"
)
codebert_ir_metrics = pd.read_csv(codebert_ir_metrics_path)
logging.info("making codebert figure")


codebert_fig = visualization_utils.make_2d_data_plot(
    reduced_features["codebert"],
    area_grouped_tasks["task"],
    area_grouped_tasks["area"],
)

st.write("# CodeBert")
st.write(
    "information retrieval metrics loaded from {}".format(codebert_ir_metrics_path)
)
st.write(codebert_ir_metrics[[col for col in lstm_ir_metrics if "cos_sim" in col]].T)
st.write(codebert_fig)
