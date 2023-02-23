import torch
import tqdm
import numpy as np
from github_search.neural_bag_of_words import tokenization


EPS = 1e-8


def fasttext_encoding_fn(fasttext_model, verbose=True, to_tensor=True):
    def encoding_fn(tokens):
        dim = fasttext_model.get_word_vector(tokens[0]).shape[0]
        embeddings_matrix = np.zeros((len(tokens), dim))
        iter = enumerate(tokens)
        if verbose:
            iter = tqdm.tqdm(iter)
        for i, w in iter:
            vector = fasttext_model.get_word_vector(w)
            # we might get a problem when we have zero vector or coordinates which are zero
            randomized_vector = np.random.normal(0, scale=1e-4, size=len(vector))
            is_coord_small = np.abs(vector) < EPS
            vector[is_coord_small] = randomized_vector[is_coord_small]
            embeddings_matrix[i] = vector / np.linalg.norm(vector)
        if to_tensor:
            return torch.Tensor(embeddings_matrix)
        else:
            return embeddings_matrix

    return encoding_fn


def get_tokenizers(document_tokenizer_path, query_tokenizer_path):

    return tokenization.TokenizerWithWeights.load(
        document_tokenizer_path
    ), tokenization.TokenizerWithWeights.load(query_tokenizer_path)


def get_ploomber_tokenizers(upstream, train_val_config):
    if "function_signature" in train_val_config.document_cols:
        tokenizer_text_col = "function_signature"
    else:
        tokenizer_text_col = "dependencies"
    tokenizer_paths = upstream["nbow.prepare_tokenizers_*"][
        f"nbow.prepare_tokenizers_{tokenizer_text_col}"
    ]
    return get_tokenizers(
        tokenizer_paths["document_tokenizer"], tokenizer_paths["query_tokenizer"]
    )
