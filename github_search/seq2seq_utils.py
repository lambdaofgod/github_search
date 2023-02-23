import ast
from functools import partial

import datasets
import pandas as pd
import transformers

PREDICT_PATH_PROMPT = "predict docID: "
TASK_SEP = "<TASK_SEP>"
PATH_TASK_SEP = "<PATH_TASK_SEP>"
REPO_NAME_SEP = "<REPO_NAME_SEP>"
REPO_PATH_SEP = "<REPO_PATH_SEP>"

SPECIAL_TOKENS = [
    TASK_SEP,
    PATH_TASK_SEP,
    REPO_NAME_SEP,
    REPO_PATH_SEP,
]


def get_path_docid_seq2seq_df(papers_with_readmes_path, python_files_path):
    paperswithcode_df = pd.read_csv(papers_with_readmes_path)
    files_df = pd.read_feather(python_files_path)
    repo_tasks = paperswithcode_df["tasks"].apply(
        lambda ts: " , ".join(ast.literal_eval(ts))
    )
    repo_tasks = pd.DataFrame({"repo": paperswithcode_df["repo"], "tasks": repo_tasks})
    files_with_tasks_df = repo_tasks.merge(files_df, on="repo")

    doc_id = (
        files_df["repo"].str.replace("/", f" {REPO_NAME_SEP} ")
        + " "
        + REPO_PATH_SEP
        + " "
        + files_df["path"]
        + " "
        + PATH_TASK_SEP
        + " "
        + files_with_tasks_df["tasks"]
    )
    seq2seq_df = pd.DataFrame(
        {"doc_id": doc_id, "contents": files_with_tasks_df["content"]}
    )
    return seq2seq_df


def load_huggingface_transformer_from_hub_or_local(model_type):
    tokenizer = transformers.RobertaTokenizerFast.from_pretrained(model_type)
    model = transformers.T5ForConditionalGeneration.from_pretrained(model_type)
    return tokenizer, model


def get_seq2seq_model_with_tokenizer(base_model):
    tokenizer, model = load_huggingface_transformer_from_hub_or_local(base_model)
    tokenizer.add_tokens(SPECIAL_TOKENS)
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer


def convert_examples_to_features(
    example_batch,
    tokenizer,
    max_source_length,
    max_target_length,
    source_col,
    target_col,
    predict_path_prompt_suffix,
):
    example_batch = example_batch.copy()

    input_encodings = tokenizer(
        example_batch[source_col],
        max_length=max_source_length,
        truncation=True,
        padding="max_length",
    )

    with tokenizer.as_target_tokenizer():
        target_encodings = tokenizer(
            example_batch[target_col],
            max_length=max_target_length,
            truncation=True,
        )

    input_ids = input_encodings["input_ids"]
    input_ids[-len(predict_path_prompt_suffix) :] = predict_path_prompt_suffix
    return {
        "input_ids": input_ids,
        "attention_mask": input_encodings["attention_mask"],
        "labels": target_encodings["input_ids"],
    }


def prepare_path_docid_seq2seq_df(
    upstream, product, max_source_length, max_target_length
):
    seq2seq_df = get_path_docid_seq2seq_df(
        upstream["make_readmes"], upstream["select_repo_files"]
    )
    source_col = "contents"
    target_col = "doc_id"
    seq2seq_df[source_col] = (
        seq2seq_df[source_col]
        .str.split()
        .apply(lambda l: " ".join(l[:max_source_length]))
    )
    seq2seq_df[target_col] = (
        seq2seq_df[target_col]
        .str.split()
        .apply(lambda l: " ".join(l[:max_target_length]))
    )

    seq2seq_df.to_csv(product, index=None)


def prepare_seq2seq_dataset(
    upstream, product, base_model, max_source_length, max_target_length
):
    seq2seq_df_path = list(upstream.values())[0]
    seq2seq_df = pd.read_csv(seq2seq_df_path)
    target_col, source_col = tuple(seq2seq_df.columns)
    seq2seq_dataset = datasets.Dataset.from_pandas(seq2seq_df)
    print(seq2seq_df.columns)
    __, tokenizer = get_seq2seq_model_with_tokenizer(
        base_model,
    )
    predict_path_prompt_suffix = tokenizer.encode(" " + PREDICT_PATH_PROMPT)

    seq2seq_dataset_pt = seq2seq_dataset.map(
        partial(
            convert_examples_to_features,
            tokenizer=tokenizer,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            source_col=source_col,
            target_col=target_col,
            predict_path_prompt_suffix=predict_path_prompt_suffix,
        ),
        batched=False,
        num_proc=10,
    )
    seq2seq_dataset_pt.save_to_disk(product)
