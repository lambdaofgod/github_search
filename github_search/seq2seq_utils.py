import transformers
import pandas as pd
import ast
from functools import partial
import datasets
from transformers import RobertaTokenizer, T5ForConditionalGeneration


TASK_SEP = "<TASK_SEP>"
PATH_TASK_SEP = "<PATH_TASK_SEP>"
REPO_NAME_SEP = "<REPO_NAME_SEP>"
REPO_PATH_SEP = "<REPO_PATH_SEP>"

SPECIAL_TOKENS = [TASK_SEP, PATH_TASK_SEP, REPO_NAME_SEP, REPO_PATH_SEP]


def get_path_docid_seq2seq_df(papers_with_readmes_path, python_files_path):
    paperswithcode_df = pd.read_csv(papers_with_readmes_path)
    files_df = pd.read_feather(python_files_path)
    repo_tasks = paperswithcode_df["tasks"].apply(
        lambda ts: TASK_SEP.join(ast.literal_eval(ts))
    )
    repo_tasks = pd.DataFrame({"repo": paperswithcode_df["repo"], "tasks": repo_tasks})
    files_with_tasks_df = repo_tasks.merge(files_df, on="repo")

    doc_id = (
        files_df["repo"].str.replace("/", REPO_NAME_SEP)
        + REPO_PATH_SEP
        + files_df["path"]
        + PATH_TASK_SEP
        + files_with_tasks_df["tasks"]
    )
    seq2seq_df = pd.DataFrame(
        {"doc_id": doc_id, "contents": files_with_tasks_df["content"]}
    )
    return seq2seq_df


def get_seq2seq_model_with_tokenizer(base_model="Salesforce/codet5-base"):
    tokenizer = RobertaTokenizer.from_pretrained(base_model)
    model = T5ForConditionalGeneration.from_pretrained(base_model)

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
):
    input_encodings = tokenizer(
        example_batch[source_col], max_length=max_source_length, truncation=True
    )

    with tokenizer.as_target_tokenizer():
        target_encodings = tokenizer(
            example_batch[target_col], max_length=max_target_length, truncation=True
        )

    return {
        "input_ids": input_encodings["input_ids"],
        "attention_mask": input_encodings["attention_mask"],
        "labels": target_encodings["input_ids"],
    }


def prepare_path_docid_seq2seq_df(upstream, product):
    seq2seq_df = get_path_docid_seq2seq_df(
        upstream["make_readmes"], upstream["select_repo_files"]
    )
    seq2seq_df.to_feather(product)


def prepare_seq2seq_dataset(
    upstream, product, base_model, max_source_length, max_target_length
):
    seq2seq_df_path = list(upstream.values())[0]
    seq2seq_df = pd.read_feather(seq2seq_df_path)
    target_col, source_col = tuple(seq2seq_df.columns)
    seq2seq_dataset = datasets.Dataset.from_pandas(seq2seq_df)

    __, tokenizer = get_seq2seq_model_with_tokenizer(
        base_model,
    )
    seq2seq_dataset_pt = seq2seq_dataset.map(
        partial(
            convert_examples_to_features,
            tokenizer=tokenizer,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            source_col=source_col,
            target_col=target_col,
        ),
        batched=True,
        batch_size=4096,
        writer_batch_size=4096,
    )
    seq2seq_dataset_pt.save_to_disk(product)
