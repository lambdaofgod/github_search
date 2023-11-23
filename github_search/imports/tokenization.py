import collections
import json
import os
import re
import string
from typing import Dict, Iterable, List, Tuple, Union

from github_search import python_tokens
from sentence_transformers.models import tokenizer


class PythonCodeTokenizer(tokenizer.WordTokenizer):
    """
    Import tokenizer to work with sentence_transformers models

    Splits by whitespace, then by '.', and then by snakecase or camelcase
    "import sklearn.decomposition" -> ["import", "sklearn", "decomposition"]
    "import sentence_transformers" -> ["import", "sentence", "transformers"]
    """

    def __init__(
        self,
        vocab: Iterable[str] = [],
        stop_words: Iterable[str] = [],
        do_lower_case: bool = False,
    ):
        self.stop_words = set(stop_words)
        self.do_lower_case = do_lower_case
        self.set_vocab(vocab)

    def get_vocab(self):
        return self.vocab

    def set_vocab(self, vocab: Iterable[str]):
        self.vocab = vocab
        self.word2idx = collections.OrderedDict(
            [(word, idx) for idx, word in enumerate(vocab)]
        )

    def get_token_iterator(self, text: str) -> Iterable[str]:
        return (
            subtoken.strip()
            for expr in text.replace("\n", " ").split()
            for token in re.split(r"\.|/", expr)
            for subtoken in python_tokens.tokenize_python(token)
            if self.is_token_a_word(subtoken.strip())
        )

    def is_token_a_word(self, token):
        number_match = re.match(r"\d+\.?\d+", token)
        return number_match is None or number_match.string != token

    def tokenize(self, text: str) -> List[int]:
        if self.do_lower_case:
            text = text.lower()

        tokens = self.get_token_iterator(text)

        tokens_filtered = []
        for token in tokens:
            if token in self.stop_words:
                continue
            elif token in self.word2idx:
                tokens_filtered.append(self.word2idx[token])
                continue

            token = token.strip(string.punctuation)
            if token in self.stop_words:
                continue
            elif len(token) > 0 and token in self.word2idx:
                tokens_filtered.append(self.word2idx[token])
                continue

            token = token.lower()
            if token in self.stop_words:
                continue
            elif token in self.word2idx:
                tokens_filtered.append(self.word2idx[token])
                continue

        return tokens_filtered

    def save(self, output_path: str):
        with open(
            os.path.join(output_path, "whitespacetokenizer_config.json"), "w"
        ) as fOut:
            json.dump(
                {
                    "vocab": list(self.word2idx.keys()),
                    "stop_words": list(self.stop_words),
                    "do_lower_case": self.do_lower_case,
                },
                fOut,
            )

    @staticmethod
    def load(input_path: str):
        with open(
            os.path.join(input_path, "whitespacetokenizer_config.json"), "r"
        ) as fIn:
            config = json.load(fIn)

        return PythonCodeTokenizer(**config)
