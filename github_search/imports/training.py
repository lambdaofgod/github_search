import logging

import pandas as pd
import sentence_transformers
import torch
from github_search import word2vec
from github_search.imports import tokenization
from github_search.sentence_embeddings import models
from sentence_transformers import losses


def train_import_word2vec(product, upstream, epochs, embedding_dim):
    tokenizer = tokenization.PythonCodeTokenizer()
    python_imports_df = pd.read_feather(str(upstream["imports.prepare_file_imports"]))
    python_imports_df["file_representation"] = (
        python_imports_df["path"] + " " + python_imports_df["imports"]
    )
    text_cols = ["file_representation"]
    word2vec_model = word2vec.train_word2vec(
        [python_imports_df],
        text_cols,
        epochs,
        embedding_dim,
        tokenize_fn=tokenizer.get_token_iterator,
    )
    word2vec.save_w2v_model(word2vec_model, str(product["binary"]), str(product["txt"]))


def train_import_rnn_file_similarity_model(
    product, upstream, rnn_config, epochs, batch_size
):
    rnn_config = models.RNNConfig(**rnn_config)
    python_imports_df = pd.read_feather(str(upstream["imports.prepare_file_imports"]))

    w2v_model = sentence_transformers.models.WordEmbeddings.from_text_file(
        str(upstream["imports.train_w2v"]["txt"])
    )
    python_imports_df["file_representation"] = (
        python_imports_df["path"] + " " + python_imports_df["imports"]
    )
    logging.info(f"building RNN model\n{rnn_config}")
    rnn = models.build_rnn_model(
        w2v_model,
        rnn_config,
    )

    train_data = models.get_input_examples(
        python_imports_df[["file_representation"]].dropna(),
        target_col="file_representation",
        source_cols=["file_representation"],
        max_seq_length=rnn_config.max_seq_length,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )

    # Use the denoising auto-encoder loss
    train_loss = losses.MultipleNegativesRankingLoss(rnn)

    rnn.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        show_progress_bar=True,
        warmup_steps=0,
        use_amp=True,
    )
    rnn.save(str(product))
