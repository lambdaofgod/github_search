# %%
import re
import shutil
from pathlib import Path as P

from github_search.neural_bag_of_words.evaluation import *

# %% tags=["parameters"]
product = None
upstream = ["nbow.train-*-*"]
params = None

# %%


def clean_number_from_name(p):
    return re.sub("-\d", "", p)


def copy_best_model(best_model_dir, dest_best_model_dir):
    if P(dest_best_model_dir).exists():
        shutil.rmtree(dest_best_model_dir)
    shutil.copytree(str(best_model_dir), dest_best_model_dir)


def rename_best_model_files(dest_best_model_dir):
    for p in P(dest_best_model_dir).glob("*"):
        cleaned_name = clean_number_from_name(str(p))
        shutil.move(str(p), cleaned_name)


def save_best_model(best_model_dir, dest_best_model_dir):
    copy_best_model(best_model_dir, dest_best_model_dir)
    rename_best_model_files(dest_best_model_dir)


# %%
yaml_paths = get_yaml_paths(upstream["nbow.train-*-*"])
metrics_df = get_metrics_df(yaml_paths).sort_values("accuracy@10", ascending=False)

# %% [markdown]
# ## Highlight top metrics
# ### NBOW evaluation metrics sorted by accuracy@10

# %%
metrics_df.style.highlight_max()

# %%
metrics_df.to_csv(product["metrics"])

# %% [markdown]
# ## Handle best model files

# %%
dest_best_model_dir = P(product["best_model_dir"])
best_model_dir = dest_best_model_dir.parent / metrics_df.index[0]
save_best_model(best_model_dir, dest_best_model_dir)


# %% [markdown]
# # Validate model files are cleaned

# %%
file_names = [
    p.name
    for p in P(dest_best_model_dir).glob("*")
    if not p.name.startswith(".")  # exclude hidden files
]
expected_file_names = ["document_nbow", "query_nbow", "validation_metrics.yaml"]
assert set(file_names) == set(expected_file_names)
