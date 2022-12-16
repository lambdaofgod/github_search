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
import yaml
from findkit import feature_extractor, index
from github_search import python_tokens, utils
from github_search.ir import evaluator
from github_search.neural_bag_of_words import *
from github_search.neural_bag_of_words import checkpointers, embedders
from github_search.neural_bag_of_words import utils as nbow_utils
from github_search.neural_bag_of_words.checkpointers import EncoderCheckpointer
from github_search.neural_bag_of_words.data import *
from github_search.neural_bag_of_words.models import *
from github_search.neural_bag_of_words.models import NBOWLayer, PairwiseNBOWModule
from github_search.neural_bag_of_words.training_utils import NBOWTrainValData
from github_search.papers_with_code import paperswithcode_tasks
from mlutil.text import code_tokenization
from nltk import tokenize
from pytorch_lightning import loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from quaterion.loss import MultipleNegativesRankingLoss
from sklearn import feature_extraction, model_selection
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from github_search.neural_bag_of_words.embedders import NBOWPair

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
upstream = []
params = None
epochs = None
batch_size = None
max_seq_length = None
loss_function_name = None
validation_metric_name = None

# %%
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

# %%


(
    train_dep_texts_with_tasks_df,
    val_dep_texts_with_tasks_df,
) = model_selection.train_test_split(df_dep_texts, test_size=0.1, random_state=0)

train_val_data = NBOWTrainValData.build(
    query_corpus,
    train_dep_texts_with_tasks_df,
    val_dep_texts_with_tasks_df,
    doc_col="dependencies",
    train_query_cols=["tasks", "titles"],
    val_query_col="tasks",
)

# %%
# Setup

fasttext_model = fasttext.load_model(fasttext_model_path)


nbow_query = NBOWLayer.make_from_encoding_fn(
    vocab=train_val_data.train_dset.get_query_vocab(),
    df_weights=train_val_data.train_dset.get_document_frequency_weights(
        train_val_data.train_dset.numericalized_queries,
        train_val_data.train_dset.get_query_vocab(),
    ),
    encoding_fn=nbow_utils.fasttext_encoding_fn(fasttext_model),
)


nbow_document = NBOWLayer.make_from_encoding_fn(
    vocab=train_val_data.train_dset.get_document_vocab(),
    df_weights=train_val_data.train_dset.get_document_frequency_weights(
        train_val_data.train_dset.numericalized_documents,
        train_val_data.train_dset.get_document_vocab(),
    ),
    encoding_fn=nbow_utils.fasttext_encoding_fn(fasttext_model),
)

nbow_pair = NBOWPair(query_nbow=nbow_query, document_nbow=nbow_document)
# # TRZEBA DODAĆ STEROWANIE LABLEKAMI

checkpointer = EncoderCheckpointer(
    train_val_data.train_dset,
    val_dep_texts_with_tasks_df,
    nbow_pair=nbow_pair,
    save_dir=product["model_dir"],
    epochs=epochs,
)

nbow_model = PairwiseNBOWModule(
    nbow_pair,
    validation_metric_name=validation_metric_name,
    checkpointer=checkpointer,
    loss_function_name=loss_function_name,
    max_len=max_seq_length,
    max_query_len=100,
    padding_value=train_val_data.train_dset.document_numericalizer.get_padding_idx(),
).to("cuda")


neptune_logger = loggers.NeptuneLogger(
    tags=["nbow", "lightning"], log_model_checkpoints=False, **neptune_args  # optional
)
neptune_logger.log_hyperparams(
    {"epochs": epochs, "max_seq_length": str(max_seq_length)}
)

# %%

trainer = pl.Trainer(
    max_epochs=max(epochs),
    accelerator="gpu",
    devices=1,
    logger=neptune_logger,
    precision=16,
    callbacks=[EarlyStopping(monitor=validation_metric_name, mode="max", patience=3)],
)


# %% [markdown]
# ## Training
# %%
trainer.fit(
    nbow_model, train_val_data.get_train_dl(batch_size), train_val_data.get_val_dl()
)

# %% [markdown]
# ## Save final model

# %%
checkpointer.unconditionally_save_epoch_checkpoint(
    nbow_pair=nbow_model.nbow_pair,
    epoch="final",
)
