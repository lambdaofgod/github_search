import ast
import logging

import numpy as np
import pandas as pd
from github_search import github_readmes, python_tokens, utils
from github_search.neural_bag_of_words import embedders, tokenization
from github_search.neural_bag_of_words.data import prepare_dependency_texts
from mlutil.text import code_tokenization
from nltk import tokenize
from mlutil_rust import code_tokenization
from clearml.automation import PipelineController
from dataclasses import dataclass

NATURAL_LANGUAGE_COLS = ["titles", "readme"]

DEPENDENCY_RECORDS_PATH = "output/dependency_records.feather"
SIGNATURES_PATH = "output/python_signatures.parquet"
TRAIN_TEST_SPLIT_PATHS = {
    "test": "output/repos_test.csv",
    "train": "output/repos_train.csv",
}

from clearml.automation.controller import PipelineDecorator
from clearml import TaskTypes


def prepare_raw_dependency_data_corpus(dependency_records_path):
    import pandas as pd
    from github_search.neural_bag_of_words.prepare_data import get_dependency_texts

    dependency_records_df = pd.read_feather(dependency_records_path)
    dep_texts = get_dependency_texts(dependency_records_df)
    dep_texts = dep_texts.dropna().rename({"destination": "dependencies"}, axis=1)
    raw_dep_texts_df = pd.DataFrame(dep_texts).reset_index()
    print("prepared raw dep texts")
    return raw_dep_texts_df


def prepare_readmes(raw_dep_texts_df, max_workers):
    import pandas as pd
    from github_search import github_readmes, python_tokens, utils

    readmes = github_readmes.get_readmes(raw_dep_texts_df, max_workers)
    dep_texts_with_readmes_df = raw_dep_texts_df.assign(readme=readmes)
    return dep_texts_with_readmes_df


def prepare_dependency_data_corpus(raw_dep_texts_df):
    dep_texts = raw_dep_texts_df  # ["dependencies"]
    print("prepared dep texts")
    return dep_texts


def prepare_rarest_signatures_corpus(signatures_path, n_rarest):
    from github_search.python_code import signatures

    rarest_signatures_corpus_pldf = (
        signatures.SignatureSelector.prepare_rarest_signatures_corpus_pldf(
            signatures_path, n_rarest
        )
    )
    signatures_corpus = rarest_signatures_corpus_pldf.to_pandas()
    print("prepared signatures")
    return signatures_corpus


def prepare_nbow_dataset(
    df_dependency_corpus,
    df_signatures_corpus,
    train_test_split_paths,
    additional_columns,
    n_readme_lines,
):
    from github_search.neural_bag_of_words.prepare_data import get_nbow_dataset
    import pandas as pd

    df_corpus_splits = {}
    for split_name in ["train", "test"]:
        df_paperswithcode = pd.read_csv(train_test_split_paths[split_name])
        df_corpus = get_nbow_dataset(
            df_paperswithcode,
            df_dependency_corpus,
            df_signatures_corpus,
            additional_columns,
            n_readme_lines,
        )
        df_corpus_splits[split_name] = df_corpus
    return df_corpus_splits


def prepare_tokenizers(upstream, product, document_col, min_freq, max_seq_length):
    logging.basicConfig(level="INFO")
    df_corpus = pd.read_parquet(str(upstream["nbow.prepare_dataset"]["train"]))
    titles = df_corpus["titles"].apply(lambda ts: " ".join(ast.literal_eval(ts)))
    tasks = df_corpus["tasks"].apply(lambda ts: " ".join(ast.literal_eval(ts)))
    query_corpus = pd.concat([titles, tasks])
    document_corpus = pd.concat([df_corpus[document_col], titles])
    logging.info("preparing query tokenizer")
    query_tokenizer = tokenization.TokenizerWithWeights.make_from_data(
        text_type="text",
        min_freq=min_freq,
        max_seq_length=max_seq_length,
        data=query_corpus,
    )
    logging.info("preparing document tokenizer")
    document_tokenizer = tokenization.TokenizerWithWeights.make_from_data(
        text_type="text" if document_col in NATURAL_LANGUAGE_COLS else "code",
        min_freq=min_freq,
        max_seq_length=max_seq_length,
        data=document_corpus,
    )
    query_tokenizer.save(str(product["query_tokenizer"]))
    document_tokenizer.save(str(product["document_tokenizer"]))


@dataclass
class NBOWPreprocessingConfig:

def setup_nbow_pipeline(
    dependency_records_path,
    signatures_path,
    train,
    n_rarest,
    train_test_split_paths,
    additional_columns=["titles"],
    n_readme_lines=10,
):
    pipe = PipelineController(
        name="Pipeline demo",
        project="examples",
        version="0.0.1",
        add_pipeline_tags=False,
    )
    pipe.add_function_step(
        "prepare_raw_dependency_data_corpus",
        function=prepare_raw_dependency_data_corpus,
        function_kwargs=dict(dependency_records_path=dependency_records_path),
        function_return=["raw_dep_texts_df"],
        cache_executed_step=True,
    )
    raw_dep_texts_df = prepare_raw_dependency_data_corpus(dependency_records_path)
    pipe.add_function_step(
        name="prepare_dependency_data_corpus",
        function=prepare_dependency_data_corpus,
        function_kwargs=dict(
            raw_dep_texts_df="${prepare_raw_dependency_data_corpus.raw_dep_texts_df}"
        ),
        function_return=["dependency_data_corpus"],
        cache_executed_step=True,
    )
    pipe.add_function_step(
        name="prepare_rarest_signature_corpus",
        function=prepare_rarest_signatures_corpus,
        function_kwargs=dict(signatures_path=signatures_path, n_rarest=n_rarest),
        function_return=["signatures_corpus"],
        cache_executed_step=True,
    )
    pipe.add_function_step(
        "prepare_nbow_dataset",
        function=prepare_nbow_dataset,
        function_kwargs=dict(
            df_dependency_corpus="${prepare_dependency_data_corpus.dependency_data_corpus}",
            df_signatures_corpus="${prepare_rarest_signature_corpus.signatures_corpus}",
            train_test_split_paths=train_test_split_paths,
            additional_columns=additional_columns,
            n_readme_lines=n_readme_lines,
        ),
        function_return=["nbow_dataset"],
        cache_executed_step=True,
    )
    return pipe


@hydra.main(config_name="nbow_pipeline", config_path="conf", version_base="1.2")
def main(cfg: DictConfig):
    data_config = cfg.data_config
    training_config = cfg.trainn

if __name__ == "__main__":
    pipe = setup_nbow_pipeline(
        DEPENDENCY_RECORDS_PATH,
        SIGNATURES_PATH,
        n_rarest=50,
        train_test_split_paths=TRAIN_TEST_SPLIT_PATHS,
    )
    pipe.set_default_execution_queue("default")
    pipe.start_locally(run_pipeline_steps_locally=True)
