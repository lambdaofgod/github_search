# %%
import ast
import itertools
import json
import pathlib
import pickle
from dataclasses import dataclass
from functools import partial
from typing import Callable, List

import dill as pickle
import fasttext
import livelossplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.utils
import torchtext
import tqdm
from findkit import feature_extractor, index
from github_search import python_tokens, utils
from github_search.ir import evaluator
from github_search.neural_bag_of_words import embedders
from github_search.neural_bag_of_words import utils as nbow_utils
from github_search.neural_bag_of_words.data import *
from github_search.neural_bag_of_words.models import *
from github_search.neural_bag_of_words.models import NBOWLayer, PairwiseNBOWModule
from github_search.neural_bag_of_words.training import *
from github_search.papers_with_code import paperswithcode_tasks
from pytorch_lightning import loggers
from quaterion.loss import MultipleNegativesRankingLoss
from sklearn import feature_extraction, model_selection
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from mlutil.text import code_tokenization
from nltk import tokenize
import yaml

# %%
# probably for deletion
hypoth_product = {
    "document_nbow": "../../output/models/document_nbow",
    "query_nbow": "../../output/models/query_nbow",
}
hypoth_upstream = {
    "nbow.prepare_dataset": {"train": "../../output/nbow_data_train.parquet"}
}
# %% tags=["parameters"]
product = None
upstream = ["nbow.prepare_dataset", "prepare_repo_train_test_split"]
params = None

# %%
epochs = epochs
max_seq_length = max_seq_length
paperswithcode_df = utils.load_paperswithcode_df(
    upstream["prepare_repo_train_test_split"]["train"]
)
df_dep_texts = pd.read_parquet(upstream["nbow.prepare_dataset"]["train"])
df_dep_texts["tasks"] = df_dep_texts["tasks"].apply(ast.literal_eval)
fasttext_model_path = str(upstream["train_python_token_fasttext"])
neptune_args = json.loads(open(neptune_config_path, "r").read())

paperswithcode_df = utils.load_paperswithcode_df("data/paperswithcode_with_tasks.csv")
query_corpus = pd.concat(
    [
        paperswithcode_df["titles"].apply(" ".join),
        paperswithcode_df["tasks"].apply(" ".join),
    ]
).str.lower()


plt.style.use("dark_background")


query_num = NBOWNumericalizer.build_from_texts(
    query_corpus.dropna(), tokenizer=tokenize.wordpunct_tokenize
)

(
    train_dep_texts_with_tasks_df,
    val_dep_texts_with_tasks_df,
) = model_selection.train_test_split(df_dep_texts, test_size=0.1, random_state=0)


document_num = NBOWNumericalizer.build_from_texts(
    train_dep_texts_with_tasks_df["dependencies"],
    tokenizer=code_tokenization.tokenize_python_code,
)


train_dset = QueryDocumentDataset.prepare_from_dataframe(
    train_dep_texts_with_tasks_df,
    ["tasks", "titles"],
    "dependencies",
    query_numericalizer=query_num,
    document_numericalizer=document_num,
)
val_dset = QueryDocumentDataset(
    val_dep_texts_with_tasks_df["tasks"].apply(" ".join).to_list(),
    val_dep_texts_with_tasks_df["dependencies"].to_list(),
    query_numericalizer=query_num,
    document_numericalizer=document_num,
)

dset = train_dset

# %%
# Setup

fasttext_model = fasttext.load_model(fasttext_model_path)


nbow_query = NBOWLayer.make_from_encoding_fn(
    vocab=dset.get_query_vocab(),
    df_weights=dset.get_document_frequency_weights(
        dset.numericalized_queries, dset.get_query_vocab()
    ),
    encoding_fn=nbow_utils.fasttext_encoding_fn(fasttext_model),
)


nbow_document = NBOWLayer.make_from_encoding_fn(
    vocab=dset.get_document_vocab(),
    df_weights=dset.get_document_frequency_weights(
        dset.numericalized_documents, dset.get_document_vocab()
    ),
    encoding_fn=nbow_utils.fasttext_encoding_fn(fasttext_model),
)
# # TRZEBA DODAĆ STEROWANIE LABLEKAMI


nbow_model = PairwiseNBOWModule(
    nbow_query, nbow_document, max_len=max_seq_length, max_query_len=100
).to("cuda")


train_dl = train_dset.get_pair_data_loader(shuffle=True)
val_dl = val_dset.get_pair_data_loader(shuffle=False, batch_size=256)


neptune_logger = loggers.NeptuneLogger(
    tags=["nbow", "lightning"], log_model_checkpoints=False, **neptune_args  # optional
)
neptune_logger.log_hyperparams({"epochs": epochs, "max_seq_length": max_seq_length})

trainer = pl.Trainer(
    max_epochs=epochs, accelerator="gpu", devices=1, logger=neptune_logger, precision=16
)

# %%
# Training
trainer.fit(nbow_model, train_dl, val_dl)
# %%
# Saving artifacts

query_embedder = embedders.make_sentence_transformer_nbow_model(
    nbow_query,
    dset.query_numericalizer.vocab.vocab.itos_,
    dset.query_numericalizer.tokenizer,
    nbow_query.token_weights.cpu().numpy().tolist(),
    max_seq_length=max_seq_length,
)


document_embedder = embedders.make_sentence_transformer_nbow_model(
    nbow_document,
    dset.document_numericalizer.vocab.vocab.itos_,
    dset.document_numericalizer.tokenizer,
    nbow_document.token_weights.cpu().numpy().tolist(),
    max_seq_length=max_seq_length,
)

query_embedder.save(product["query_nbow"])
document_embedder.save(product["document_nbow"])


# %%
# Metrics

ir_evaluator = evaluator.InformationRetrievalEvaluator(
    document_embedder, query_embedder
)
ir_evaluator.setup(val_dep_texts_with_tasks_df.reset_index(), "tasks", "dependencies")
metrics = ir_evaluator.evaluate()

print("information retrieval metrics")
print(yaml.dump(metrics["cos_sim"]))

with open(str(product["metrics"]), "w") as f:
    yaml.dump(metrics, f)
