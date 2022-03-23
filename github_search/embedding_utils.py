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


rnn_model_path = "output/sbert/rnn2x256_epoch325/"
codebert_model_path = "output/sbert/codebert15/"


def prepare_reduced_embeddings(
    product,
    upstream,
    rnn_model_path="output/sbert/rnn2x256_epoch325/",
    codebert_model_path="output/sbert/codebert15/",
):
    w2v_model_path = str(upstream["train_abstract_readme_w2v"])
    fasttext_model_path = str(upstream["train_python_token_fasttext"])
    logging.info("preparing ")
    fasttext_model = fasttext.load_model(fasttext_model_path)
    codebert_model = sentence_transformers.SentenceTransformer(codebert_model_path)
    rnn_model = sentence_transformers.SentenceTransformer(rnn_model_path)
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
    logging.info("preparing rnn features")
    reduced_features["rnn"] = feature_extraction_utils.get_reduced_embeddings_df(
        area_grouped_tasks["task"],
        embeddings.SentenceTransformerWrapper(rnn_model),
        umap.UMAP(metric="cosine"),
    )
    logging.info("preparing codebert features")
    reduced_features["codebert"] = feature_extraction_utils.get_reduced_embeddings_df(
        area_grouped_tasks["task"],
        embeddings.SentenceTransformerWrapper(codebert_model),
        umap.UMAP(metric="cosine"),
    )
    pickle.dump(reduced_features, open(str(product), "wb"))
