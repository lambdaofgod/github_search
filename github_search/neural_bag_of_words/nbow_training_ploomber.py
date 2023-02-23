# %%
import os
import json

import fasttext
import numpy as np
import pandas as pd
import torch
import tqdm
import pytorch_lightning as pl
from github_search import utils
from github_search.neural_bag_of_words import *
from github_search.neural_bag_of_words import checkpointers, embedders, training_utils
from github_search.neural_bag_of_words import utils as nbow_utils
from github_search.neural_bag_of_words.checkpointers import EncoderCheckpointer
from github_search.neural_bag_of_words.data import *
from github_search.neural_bag_of_words.pairwise_models import PairwiseEmbedderModule
from github_search.neural_bag_of_words.training_utils import (
    TrainValConfig,
    NBOWTrainValData,
    QueryDocumentCollator,
)
from github_search.neural_bag_of_words.embedders import (
    EmbedderFactory,
    ModelPairConfig,
)
from pytorch_lightning import loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn import feature_extraction, model_selection
import logging

logging.basicConfig(level="INFO")

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
upstream = {}
params = None
epochs = None
batch_size = None
validation_metric_name = None
config_name = None

# %% [markdown]
# ## Load config
# %%
train_val_config = TrainValConfig.load(config_name)
max_seq_length = train_val_config.max_seq_length

# %% [markdown]
# ## Load data
# %%

paperswithcode_df = utils.load_paperswithcode_df(
    upstream["prepare_repo_train_test_split"]["train"]
)
df_dep_texts = pd.read_parquet(upstream["nbow.prepare_dataset"]["train"])
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
# %% [markdown]
# ## Prepare train and validation data
# %%
train_val_data = NBOWTrainValData.build(
    query_corpus,
    train_dep_texts_with_tasks_df,
    val_dep_texts_with_tasks_df,
    train_val_config,
)

# %% [markdown]
# ## Setup
# Tokenizers
# %%

document_tokenizer, query_tokenizer = nbow_utils.get_tokenizers_ploomber(
    upstream, train_val_config
)

# %% [markdown]
# Embedders
# %%


def get_fasttext_encoding_fn(fasttext_path):
    fasttext_model = fasttext.load_model(fasttext_path)
    return nbow_utils.fasttext_encoding_fn(fasttext_model)


fasttext_encoding_fn = get_fasttext_encoding_fn(fasttext_model_path)


embedder_pair = EmbedderFactory.make_from_train_val_config(
    train_val_config,
    fasttext_encoding_fn,
    max_seq_length,
    query_tokenizer,
    document_tokenizer,
).get_embedder_pair()

collator = QueryDocumentCollator.from_embedder_pair(embedder_pair)

checkpointer = EncoderCheckpointer(
    train_val_data.train_dset,
    val_dep_texts_with_tasks_df,
    embedder_pair=embedder_pair,
    column_config=train_val_config.get_information_retrieval_column_config(),
    save_dir=product["model_dir"],
    epochs=epochs,
)

# %%

nbow_model = PairwiseEmbedderModule(
    embedder_pair=embedder_pair,
    validation_metric_name=validation_metric_name,
    checkpointer=checkpointer,
    loss_function_name=train_val_config.loss_function_name,
    max_len=max_seq_length,
    max_query_len=100,
    train_query_embedder=train_val_config.train_query_embedder,
    shuffle_documents=train_val_config.shuffle_documents,
).to("cuda")

# %%

tags = ["nbow", "lightning"] + train_val_config.document_cols
if train_val_config.query_embedder is not None:
    tags.append(train_val_config.query_embedder)

print(tags)

# %%

neptune_logger = loggers.NeptuneLogger(
    tags=tags, log_model_checkpoints=False, **neptune_args  # optional
)
neptune_logger.log_hyperparams(
    {"epochs": epochs, "max_seq_length": str(max_seq_length), "conf_name": config_name}
)
neptune_logger.experiment["config"].upload(os.path.join("conf", config_name))

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

logging.info("starting training...")

trainer.fit(
    nbow_model,
    train_val_data.get_train_dl(batch_size=batch_size, collator=collator),
    train_val_data.get_val_dl(batch_size=256, collator=collator),
)

# %% [markdown]
# ## Save final model

# %%

checkpointer.unconditionally_save_epoch_checkpoint(
    embedder_pair=nbow_model.embedder_pair,
    epoch="final",
)
