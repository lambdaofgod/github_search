import logging
import re

import gensim.models
import pandas as pd
from gensim.models.callbacks import CallbackAny2Vec
from typing import List, Callable, Iterable
from mlutil.text import code_tokenization
from github_search import paperswithcode_tasks


def clean_whitespaces(s):
    return re.sub(r"\s+", " ", s)


def default_tokenize_fn(text):
    return text.split()


def get_sentences(
    dfs: List[pd.DataFrame],
    text_cols: List[str],
    tokenize_fn: Callable[[str], Iterable[str]],
    max_length: int = 1000,
):
    text_series = (
        df[text_col].dropna().apply(tokenize_fn)
        for (df, text_col) in zip(dfs, text_cols)
    )
    return (list(sent)[:max_length] for texts in text_series for sent in texts)


class LossLogger(CallbackAny2Vec):
    """Output loss at each epoch"""

    def __init__(self):
        self.epoch = 1
        self.losses = []

    def on_epoch_begin(self, model):
        print(f"Epoch: {self.epoch}", end="\t")

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        self.losses.append(loss)
        print(f"  Loss: {loss}")
        self.epoch += 1


class LossCallback(CallbackAny2Vec):
    """
    Callback to print loss after each epoch
    """

    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        if self.epoch == 0:
            print("Loss after epoch {}: {}".format(self.epoch, loss))
        else:
            print(
                "Loss after epoch {}: {}".format(
                    self.epoch, loss - self.loss_previous_step
                )
            )
        self.epoch += 1
        self.loss_previous_step = loss


def make_w2v_model(sentences, embedding_dim=200):
    w2v_model = gensim.models.Word2Vec(
        size=embedding_dim,
        window=5,
        min_count=5,
        workers=24,
        callbacks=[LossCallback()],
    )
    w2v_model.build_vocab(sentences, progress_per=1000)
    return w2v_model


def train_word2vec(
    dfs,
    text_cols,
    epochs,
    embedding_dim,
    tokenize_fn: Callable[[str], Iterable[str]] = default_tokenize_fn,
):
    sentences = get_sentences(dfs, text_cols, tokenize_fn=tokenize_fn)
    if epochs > 1:
        sentences = list(sentences)

    w2v_model = make_w2v_model(sentences, embedding_dim)
    if epochs == 1:
        sentences = get_sentences(dfs, text_cols, tokenize_fn=tokenize_fn)
    w2v_model.train(
        sentences,
        total_examples=w2v_model.corpus_count,
        epochs=epochs,
        report_delay=1,
        compute_loss=True,
    )
    return w2v_model


def save_w2v_model(w2v_model, bin_path, word2vec_path):
    if bin_path is not None:
        w2v_model.save(bin_path)
    if word2vec_path is not None:
        w2v_model.wv.save_word2vec_format(word2vec_path)


def train_abstract_readme_w2v(embedding_dim, epochs, upstream, product):
    paperswithcode_df, all_papers_df = paperswithcode_tasks.get_paperswithcode_dfs()
    papers_with_readmes_df = pd.read_csv(upstream["make_readmes"])
    word2vec_model = train_word2vec(
        [all_papers_df, papers_with_readmes_df],
        ["abstract", "readme"],
        epochs,
        embedding_dim,
    )
    save_w2v_model(word2vec_model, str(product["binary"]), str(product["txt"]))


def train_python_code_w2v(python_file_path, embedding_dim, product):
    python_code_df = pd.read_feather(
        python_file_path.replace("parquet", "feather"), columns=["content"]
    )
    word2vec_model = train_word2vec(
        [python_code_df],
        ["content"],
        1,
        embedding_dim,
        tokenize_fn=code_tokenization.tokenize_python_code,
    )
    save_w2v_model(word2vec_model, str(product["binary"]), str(product["txt"]))
