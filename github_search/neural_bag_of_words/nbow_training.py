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
    NBOWModelConfig,
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
import clearml
from dataclasses import asdict, dataclass
from github_search.neural_bag_of_words.configs import *

logging.basicConfig(level="INFO")
# %%
# probably for deletion
# %% tags=["parameters"]


@dataclass
class TokenizerConfig:
    document_tokenizer_path: str
    query_tkenizer_path: str


@dataclass
class LoggerArgs:
    neptune_config_path: str


@dataclass
class TrainingConfig:
    query_config: QueryFeatureConfig
    document_config: DocumentFeatureConfig
    hyperparameter_config: HyperparameterConfig
    logger_config: LoggerConfig


def setup_logger(config_name, logger_args: LoggerArgs):
    with open(logger_args.neptune_config_path, "r") as f:
        neptune_args = json.loads(f.read())
    neptune_logger = loggers.NeptuneLogger(
        tags=tags, log_model_checkpoints=False, **neptune_args  # optional
    )
    neptune_logger.log_hyperparams(
        {
            "epochs": epochs,
            "max_seq_length": str(max_seq_length),
            "conf_name": config_name,
        }
    )
    neptune_logger.experiment["config"].upload(os.path.join("conf", config_name))


def train(nbow_dataset, training_config: TrainingConfig):
    epochs = training_config.epochs
    batch_size = training_config.batch_size
    validation_metric_name = training_config.validation_metric_name
    # %% [markdown]
    # ## Load config
    # %%
    train_val_config = NBOWModelConfig.load(training_config.model_config_name)
    max_seq_length = train_val_config.max_seq_length

    # %% [markdown]
    # ## Load data
    # %%

    paperswithcode_df = utils.load_paperswithcode_df(
        training_config.paperswithcode_train_path
    )
    df_dep_texts = nbow_dataset  # pd.read_parquet(training_config.training_data_path)
    fasttext_model_path = str(training_config.fasttext_path)

    paperswithcode_df = utils.load_paperswithcode_df(
        training_config.paperswithcode_with_tasks_path
    )
    query_corpus = pd.concat(
        [
            paperswithcode_df["titles"].apply(" ".join),
            paperswithcode_df["tasks"].apply(" ".join),
        ]
    ).str.lower()
    logger = setup_logger(training_config.logger_args)

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

    document_tokenizer, query_tokenizer = nbow_utils.get_tokenizers(
        document_tokenizer=training_config.document_config.tokenizer_path,
        query_tokenizer=training_config.query_config.tokenizer_path,
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

    # %%

    trainer = pl.Trainer(
        max_epochs=max(epochs),
        accelerator="gpu",
        devices=1,
        logger=logger,
        precision=16,
        callbacks=[
            EarlyStopping(monitor=validation_metric_name, mode="max", patience=3)
        ],
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
