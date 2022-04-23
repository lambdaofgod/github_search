import ast
import logging
import pathlib

import numpy as np
import pandas as pd
from sentence_transformers import losses
from torch.utils.data import DataLoader

from github_search import paperswithcode_tasks, sentence_embeddings
from github_search.sentence_embeddings import RNN_MODEL_TYPES

logging.basicConfig(level="INFO")

# %%
np.random.seed(1)

# + tags=["parameters"]
upstream = None
product = None
w2v_file = "output/abstract_readme_w2v200.txt"
model_type = (
    "ir_model/checkpoint-300"
    # "stmnk/codet5-small-code-summarization-python"
    # "Salesforce/codet5-small-code-summarization-python"
    # "SEBIS/code_trans_t5_small_source_code_summarization_python_multitask_finetune"
)
num_layers = 2
n_hidden = 192
dropout = 0.25
batch_size = 128 
use_amp = "lstm" not in model_type
n_epochs = 50
show_results_n_epochs = 5
train_target_col = "tasks"
training_run_label = "imports+comments"
train_source_cols = ["repo"]  # "title", "abstract", "readme"]
df_path = "output/selected_python_files_comments_imports.csv"
paperswithcode_filepath = "output/papers_with_readmes.csv"
max_seq_length = 64
pooling_kwargs = dict(
    pooling_mode_mean_tokens=False,
    pooling_mode_cls_token=False,
    pooling_mode_max_tokens=True,
)
train_feature_col = "repo"
use_imports_as_doc = train_feature_col == "imports"
# %% [markdown]
# ## Setting up variables

# %%
if any(rnn_type in model_type for rnn_type in RNN_MODEL_TYPES):
    logging.info("training RNN model")
    logging.info("initializing word embeddings from %s", w2v_file)
    logging.info({"num_layers": num_layers, "n_hidden": n_hidden, "dropout": dropout})
    model_type_ext = model_type + str(num_layers) + "x" + str(n_hidden)
else:
    logging.info("training %s transformer model", model_type)
    model_type_ext = model_type

paperswithcode_df = pd.read_csv(paperswithcode_filepath).dropna(
    subset=["least_common_task", "readme", "abstract"]
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


def get_sbert_inputs(train_df, train_target_col, train_source_cols, model):
    train_df = train_df.dropna(subset=[train_target_col] + train_source_cols)
    # train_df[train_target_col] = train_df[train_target_col].apply(
    #    lambda examples: [examples] if type(examples) is not list else examples
    # )

    return sentence_embeddings.get_input_examples(
        train_df,
        model=model,
        target_col=train_target_col,
        source_cols=train_source_cols,
        max_seq_length=max_seq_length,
    )


def get_merged_import_df(paperswithcode_df, imports_df):
    repo_imports_df = (
        imports_df.dropna()[["repo", "imports"]].groupby("repo").agg("\n".join)
    )
    return paperswithcode_df.merge(repo_imports_df, on=["repo"])


# %% [markdown]
# ## Training loop

# %%

# text_df = pd.DataFrame(ex.texts for ex in train_input_examples)
# len_df = pd.DataFrame([len(t) for t in ex.texts] for ex in train_input_examples)

if __name__ == "__main__":

    model = sentence_embeddings.make_model(
        model_type,
        w2v_file,
        num_layers=num_layers,
        n_hidden=n_hidden,
        dropout=dropout,
        max_seq_length=max_seq_length,
        **pooling_kwargs
    )

    repo_imports_df = get_merged_import_df(paperswithcode_df, pd.read_csv(df_path))
    logging.info("loading readme-imports examples")
    # imports_train_input_examples = get_sbert_inputs(
    #    repo_imports_df, train_target_col, train_source_cols, model
    # )
    logging.info("loading task-readme examples")
    tasks_train_input_examples = get_sbert_inputs(
        paperswithcode_df[["tasks", train_feature_col]].explode("tasks"),
        "tasks",
        [train_feature_col],
        model,
    )
    train_input_examples = tasks_train_input_examples
    print([i.texts for i in train_input_examples[:5]])
    paperswithcode_df = pd.read_csv(paperswithcode_filepath)
    use_imports_as_doc = False
    if use_imports_as_doc:
        test_df = get_merged_import_df(paperswithcode_df, pd.read_csv(df_path))
        ir_evaluator = sentence_embeddings.get_ir_evaluator(test_df, "tasks", "imports")
    else:
        ir_evaluator = sentence_embeddings.get_ir_evaluator(
            paperswithcode_df, doc_col=train_feature_col
        )

    logging.info("loaded %s train samples", int(len(train_input_examples)))

    train_dataloader = DataLoader(
        train_input_examples, shuffle=True, batch_size=batch_size
    )

    train_loss = losses.MultipleNegativesSymmetricRankingLoss(model)

    for epoch_iter in range(0, n_epochs, show_results_n_epochs):
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

        model_dir_suffix = (
            model_type_ext.replace("/", "_")
            + "_epoch"
            + str(epoch_iter + show_results_n_epochs)
        )
        model_path = "output/sbert/{}_{}".format(model_dir_suffix, training_run_label)
        pathlib.Path(model_path).mkdir(parents=True)
        model.save(model_path)

        ir_evaluator(model, output_path="output/sbert/" + model_dir_suffix)
        # print(get_ir_metrics("output/sbert/" + model_dir_suffix))
