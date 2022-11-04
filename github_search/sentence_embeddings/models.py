import os

import pandas as pd
import tqdm
from mlutil.sentence_rnn import SentenceRNN
from sentence_transformers import InputExample, SentenceTransformer, evaluation, models
from typing import Optional

from github_search.ir.ir_utils import get_ir_evaluator
from github_search.imports import tokenization
from dataclasses import dataclass


RNN_MODEL_TYPES = ["sru", "lstm"]


tokenizer_mapping = {
    "WhitespaceTokenizer": models.tokenizer.WhitespaceTokenizer,
    "PythonCodeTokenizer": tokenization.PythonCodeTokenizer,
}


@dataclass
class RNNConfig:
    model_type: str
    pooling_mode_mean_tokens: bool
    pooling_mode_max_tokens: bool
    num_layers: int = 2
    n_hidden: int = 256
    dropout: float = 0.25
    max_seq_length: int = 256
    tokenizer: Optional[str] = None

    def get_tokenizer(self):
        return tokenizer_mapping[self.tokenizer]()


def get_input_examples(df, target_col, source_cols, max_seq_length):
    sources = [df[src_col] for src_col in source_cols]
    src_lengths_masks = [pd.Series(src.str.split().apply(len) > 0) for src in sources]
    target_source_pairs = [
        (target_text, source_text)
        for (mask, source) in zip(src_lengths_masks, sources)
        for (target_text, source_text) in zip(df[target_col][mask], source[mask])
    ]
    return [
        InputExample(
            texts=[
                " ".join(target.split()[:max_seq_length]),
                " ".join(source.split()[:max_seq_length]),
            ]
        )
        for (target, source) in tqdm.tqdm(target_source_pairs)
    ]


def _init_tokenizer_from_others_vocab(tokenizer, vocab):
    if len(tokenizer.get_vocab()) == 0:
        tokenizer.set_vocab(vocab)
    return tokenizer


def build_rnn_model(word_embedding_model, rnn_config: RNNConfig) -> SentenceTransformer:
    rnn_model_types = ["lstm", "sru"]
    maybe_chosen_model_type = [
        tpe for tpe in rnn_model_types if tpe in rnn_config.model_type
    ]
    assert len(maybe_chosen_model_type) == 1
    [chosen_model_type] = maybe_chosen_model_type

    rnn = SentenceRNN(
        word_embedding_dimension=word_embedding_model.get_word_embedding_dimension(),
        dropout=rnn_config.dropout,
        hidden_dim=rnn_config.n_hidden,
        num_layers=rnn_config.num_layers,
        rnn_class_type=chosen_model_type,
    )
    if rnn_config.tokenizer is not None:
        tokenizer = rnn_config.get_tokenizer()
        tokenizer = _init_tokenizer_from_others_vocab(
            tokenizer, word_embedding_model.tokenizer.vocab
        )
        word_embedding_model.tokenizer = tokenizer

    pooling_model = models.Pooling(
        rnn.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=rnn_config.pooling_mode_mean_tokens,
        pooling_mode_max_tokens=rnn_config.pooling_mode_max_tokens,
        pooling_mode_cls_token=False,
    )
    model = SentenceTransformer(modules=[word_embedding_model, rnn, pooling_model])
    return model


def make_model(
    model_type,
    w2v_file,
    num_layers,
    n_hidden,
    dropout,
    max_seq_length,
    is_model_sentence_transformer,
    pooling_mode_mean_tokens,
    pooling_mode_cls_token,
    pooling_mode_max_tokens,
):
    if any(rnn_type in model_type for rnn_type in RNN_MODEL_TYPES):
        word_embedding_model = models.WordEmbeddings.from_text_file(w2v_file)
        model = make_rnn_model(
            word_embedding_model=word_embedding_model,
            model_type=model_type,
            num_layers=num_layers,
            n_hidden=n_hidden,
            dropout=dropout,
            max_seq_length=max_seq_length,
            pooling_mode_mean_tokens=pooling_mode_mean_tokens,
            pooling_mode_max_tokens=pooling_mode_max_tokens,
        )
    elif is_model_sentence_transformer:
        model = SentenceTransformer(model_type)
    else:
        transformer = models.Transformer(model_type, max_seq_length=max_seq_length)
        pooling_model = models.Pooling(
            transformer.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=pooling_mode_mean_tokens,
            pooling_mode_max_tokens=pooling_mode_max_tokens,
        )
        model = SentenceTransformer(modules=[transformer, pooling_model])
    return model


def build_word_embeddings_sentence_transformer_model(
    word_embeddings,
    pooling_mode_mean_tokens: bool = True,
    pooling_mode_max_tokens: bool = False,
):
    pooling_model = models.Pooling(
        word_embeddings.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=pooling_mode_mean_tokens,
        pooling_mode_max_tokens=pooling_mode_max_tokens,
    )
    return SentenceTransformer(
        modules=[word_embeddings, pooling_model, models.Normalize()]
    )


def build_word_embeddings_sentence_transformer_model_from_file(
    word_embeddings_file: str,
    pooling_mode_mean_tokens: bool = True,
    pooling_mode_max_tokens: bool = False,
) -> SentenceTransformer:
    """
    build from word embeddings from word_embeddings_file
    word_embeddings_file is assumed to be a word2vec .txt format file
    """
    word_embeddings = models.WordEmbeddings.from_text_file(word_embeddings_file)

    return build_word_embeddings_sentence_transformer_model(
        word_embeddings, pooling_mode_mean_tokens, pooling_mode_max_tokens
    )


def prepare_word2vec_sentence_embedding_model(upstream, product):
    word2vec_path = str(upstream["train_abstract_readme_w2v"]).replace("bin", "txt")
    model = build_word_embeddings_sentence_transformer_model_from_file(word2vec_path)
    model.save(str(product))
