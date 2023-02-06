from dataclasses import dataclass, field
from typing import List, Callable, Optional, Union

import numpy as np
import os
import pickle
from mlutil import sentence_transformers_utils
import tqdm
from collections import Counter
from nltk import tokenize
from mlutil_rust import code_tokenization


MAX_DOC_CHAR_LENGTH = 10 ** 5


@dataclass
class TokenizerWithWeights:

    tokenizer: sentence_transformers_utils.CustomTokenizer
    frequency_weights: np.ndarray
    max_seq_length: int

    @classmethod
    def get_tokenize_fn(cls, text_type):
        if text_type == "code":
            return tokenize.wordpunct_tokenize
        else:
            return code_tokenization.tokenize_python_code

    @classmethod
    def make_from_data(
        cls,
        text_type: str,
        min_freq: int,
        data: List[str],
        max_seq_length: int,
    ):
        tokenize_fn = cls.get_tokenize_fn(text_type)
        token_counter = cls.get_token_counter(data, tokenize_fn, max_seq_length)
        vocab = cls.get_vocab_from_counter(token_counter, min_freq)
        tokenizer = cls.get_tokenizer(vocab, tokenize_fn)
        frequency_weights = cls.get_frequency_weights(token_counter, vocab)
        return TokenizerWithWeights(
            tokenizer=tokenizer,
            frequency_weights=frequency_weights,
            max_seq_length=max_seq_length,
        )

    @classmethod
    def get_token_counter(cls, data, tokenize_fn, max_seq_length):
        all_tokens = (
            tok.lower()
            for doc in tqdm.auto.tqdm(data)
            for tok in tokenize_fn(doc[-MAX_DOC_CHAR_LENGTH:])[:max_seq_length]
        )
        return Counter(all_tokens)

    @classmethod
    def get_vocab_from_counter(cls, counter, min_freq):
        return [item for (item, cnt) in counter.items() if cnt >= min_freq]

    @classmethod
    def get_tokenizer(cls, vocab, tokenize_fn):
        return sentence_transformers_utils.CustomTokenizer(
            vocab=vocab, tokenize_fn=tokenize_fn, do_lower_case=True
        )

    @classmethod
    def get_frequency_weights(cls, counter, vocab):
        freqs = np.array([counter.get(t, 0) for t in vocab])
        return freqs

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj
