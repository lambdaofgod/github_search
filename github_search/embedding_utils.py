import logging
import pickle

import fasttext
import gensim
import pandas as pd
import sentence_transformers
import umap
from mlutil import feature_extraction_utils
from mlutil.feature_extraction import embeddings

from github_search import paperswithcode_tasks

logging.basicConfig(level="INFO")
area_grouped_tasks = pd.read_csv("data/paperswithcode_tasks.csv").dropna()


area_grouped_tasks["task"] = area_grouped_tasks["task"].apply(
    paperswithcode_tasks.clean_task_name
)


lstm_model_path = "output/sbert/lstm2x256_epoch325/"
codebert_model_path = "output/sbert/codebert15/"


def prepare_reduced_embeddings(
    product,
    w2v_model_path="output/abstract_readme_w2v200.bin",
    fasttext_model_path="output/python_files_fasttext_dim200.bin",
    lstm_model_path="output/sbert/lstm2x256_epoch325/",
    codebert_model_path="output/sbert/codebert15/",
):
    logging.info("preparing ")
    fasttext_model = fasttext.load_model(fasttext_model_path)
    codebert_model = sentence_transformers.SentenceTransformer(codebert_model_path)
    lstm_model = sentence_transformers.SentenceTransformer(lstm_model_path)
    fasttext_embedder = embeddings.FastTextVectorizer(fasttext_model)
    w2v_embedder = embeddings.AverageWordEmbeddingsVectorizer(
        word_embeddings=gensim.models.KeyedVectors.load(w2v_model_path).wv
    )
    reduced_features = {}

    logging.info("preparing w2v features")
    reduced_features["word2vec"] = feature_extraction_utils.get_reduced_embeddings_df(
        area_grouped_tasks["task"], w2v_embedder, umap.UMAP(metric="cosine")
    )
    logging.info("preparing fasttext features")
    reduced_features["fasttext"] = feature_extraction_utils.get_reduced_embeddings_df(
        area_grouped_tasks["task"], fasttext_embedder, umap.UMAP(metric="cosine")
    )
    logging.info("preparing lstm features")
    reduced_features["lstm"] = feature_extraction_utils.get_reduced_embeddings_df(
        area_grouped_tasks["task"],
        embeddings.SentenceTransformerWrapper(lstm_model),
        umap.UMAP(metric="cosine"),
    )
    logging.info("preparing codebert features")
    reduced_features["codebert"] = feature_extraction_utils.get_reduced_embeddings_df(
        area_grouped_tasks["task"],
        embeddings.SentenceTransformerWrapper(codebert_model),
        umap.UMAP(metric="cosine"),
    )
    pickle.dump(reduced_features, open(str(product), "wb"))
