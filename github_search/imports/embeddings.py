from github_search import word2vec
import pandas as pd


def train_import_word2vec(product, upstream, epochs, embedding_dim):
    python_imports_df = pd.read_feather(str(upstream["imports.prepare_file_imports"]))
    text_cols = ["imports"]
    word2vec.train_word2vec([python_imports_df], text_cols, epochs, embedding_dim, str(product))
