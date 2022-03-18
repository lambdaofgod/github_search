import ast
import os

import attr
import numpy as np
import pandas as pd
import tqdm

np.random.seed(1)


from sentence_transformers import (
    InputExample,
    SentenceTransformer,
    evaluation,
    losses,
    models,
)
from sklearn import model_selection
from torch.utils.data import DataLoader

from github_search import paperswithcode_tasks
import logging


logging.basicConfig(level="INFO")


# + tags=["parameters"]
upstream = None
product = None
w2v_file = "output/abstract_readme_w2v200.txt"
model_type = "lstm"#microsoft/codebert-base"
num_layers = 2
n_hidden = 256
dropout = 0.25
batch_size = 128
use_amp = "lstm" not in model_type
n_epochs = 500
show_results_n_epochs = 50
paperswithcode_filepath = "output/papers_with_readmes.csv"
max_seq_length = 128
pooling_mode_mean_tokens = False
pooling_mode_cls_token = False
pooling_mode_max_tokens = True

# %% [markdown]
# ## Setting up variables


# %%
if "lstm" in model_type:
    logging.info("training LSTM model")
    logging.info("initializing word embeddings from {}".format(w2v_file))
    logging.info({"num_layers": num_layers, "n_hidden": n_hidden, "dropout": dropout})
    model_type_ext = model_type + str(num_layers) + "x" + str(n_hidden)
else:
    logging.info("training {} transformer model".format(model_type))
    model_type_ext = model_type

paperswithcode_df = pd.read_csv(paperswithcode_filepath).dropna(
    subset=["least_common_task", "abstract"]
)
paperswithcode_df["tasks"] = paperswithcode_df["tasks"].apply(ast.literal_eval)

repo_counts = paperswithcode_df["repo"].value_counts()

area_grouped_tasks = pd.read_csv("data/paperswithcode_tasks.csv").dropna()

tasks_test = pd.read_csv("output/test_tasks.csv")

area_grouped_tasks.index = area_grouped_tasks["task"].apply(
    paperswithcode_tasks.clean_task_name
)

paperswithcode_df["areas"] = paperswithcode_df["tasks"].apply(
    lambda tasks: set(
        area_grouped_tasks.loc[[t for t in tasks if t in area_grouped_tasks.index]][
            "area"
        ].tolist()
    )
)


train_df = paperswithcode_df[
    paperswithcode_df["tasks"].apply(
        lambda ts: not (any(t in tasks_test.values for t in ts))
    )
]
test_df = paperswithcode_df[
    paperswithcode_df["tasks"].apply(lambda ts: any(t in tasks_test.values for t in ts))
]

# %%
def get_sbert_ir_dicts(input_df, query_col="tasks", doc_col="abstract"):
    df_copy = input_df.copy()
    queries = df_copy[query_col].explode().drop_duplicates()
    queries = pd.DataFrame(
        {"query": queries, "query_id": [str(s) for s in queries.index]}
    )
    queries.index = queries["query_id"]
    corpus = df_copy["abstract"]
    corpus.index = [str(i) for i in corpus.index]
    df_copy["doc_id"] = corpus.index
    relevant_docs_str = df_copy[["doc_id", "tasks", doc_col]].explode(column="tasks")
    relevant_docs = (
        relevant_docs_str.merge(queries, left_on="tasks", right_on="query")[
            ["doc_id", "query_id"]
        ]
        .groupby("query_id")
        .apply(lambda df: set(df["doc_id"]))
        .to_dict()
    )
    return queries["query"].to_dict(), corpus.to_dict(), relevant_docs


def get_ir_metrics(path):
    metrics_df = pd.read_csv(
        os.path.join(path, "Information-Retrieval_evaluation_results.csv")
    )
    return metrics_df[[col for col in metrics_df if "cos" in col]]


def get_input_examples(df, aug=None, text_cols=["title", "abstract"]):
    return [
        InputExample(texts=[task, text])
        for row in tqdm.tqdm(
            df[["tasks"] + text_cols].itertuples(index=False), total=len(df)
        )
        for task in row[0]
        for text in row[1:]
    ]


def make_lstm_model(word_embedding_model, num_layers=2, n_hidden=256, dropout=0.25):
    lstm = models.LSTM(
        word_embedding_dimension=word_embedding_model.get_word_embedding_dimension(),
        dropout=dropout,
        hidden_dim=n_hidden,
        num_layers=num_layers,
    )

    pooling_model = models.Pooling(
        lstm.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=pooling_mode_mean_tokens,
        pooling_mode_cls_token=pooling_mode_cls_token,
        pooling_mode_max_tokens=pooling_mode_max_tokens,
    )
    model = SentenceTransformer(modules=[word_embedding_model, lstm, pooling_model])
    return model


def make_model(model_type, w2v_file, num_layers, n_hidden, dropout, max_seq_length):
    if "lstm" in model_type:
        word_embedding_model = models.WordEmbeddings.from_text_file(w2v_file)
        model = make_lstm_model(word_embedding_model, num_layers, n_hidden, dropout)
    else:
        transformer = models.Transformer(model_type, max_seq_length=max_seq_length)
        pooling_model = models.Pooling(
            transformer.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=pooling_mode_mean_tokens,
            pooling_mode_cls_token=pooling_mode_cls_token,
            pooling_mode_max_tokens=pooling_mode_max_tokens,
        )
        model = SentenceTransformer(modules=[transformer, pooling_model])
    return model


# %% [markdown]
# ## Training loop

# %%

queries, corpus, relevant_docs = get_sbert_ir_dicts(test_df)
ir_evaluator = evaluation.InformationRetrievalEvaluator(
    queries, corpus, relevant_docs, main_score_function="cos_sim", map_at_k=[10]
)


train_input_examples = get_input_examples(train_df, text_cols=["title", "abstract"])
test_input_examples = get_input_examples(test_df, text_cols=["abstract"])


model = make_model(model_type, w2v_file, num_layers, n_hidden, dropout, max_seq_length)
train_dataloader = DataLoader(train_input_examples, shuffle=True, batch_size=batch_size)


train_loss = losses.MultipleNegativesRankingLoss(model)


for epoch_iter in range(0, n_epochs + 1, show_results_n_epochs):
    logging.info("#" * 100)
    logging.info("#" * 100)
    logging.info("EPOCH {}".format(str(epoch_iter)))
    logging.info("#" * 100)
    logging.info("#" * 100)
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=show_results_n_epochs,
        show_progress_bar=True,
        evaluator=ir_evaluator,
        evaluation_steps=1000,
        use_amp=use_amp,
        callback=lambda score, epoch, steps: print(
            "epoch {}, step {}, map@10: {}".format(epoch, steps, round(score, 3))
        ),
    )

    model_dir_suffix = model_type_ext.replace("/", "_") + "_epoch" + str(epoch_iter)
    model.save("output/sbert/" + model_dir_suffix)

    ir_evaluator(model, output_path="output/sbert/" + model_dir_suffix)
    # print(get_ir_metrics("output/sbert/" + model_dir_suffix))
