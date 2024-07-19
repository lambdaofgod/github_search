# +
import pandas as pd
import pickle
from pathlib import Path
import os

from github_search.utils import return_result


# -


def get_filename_without_extension(p):
    try:
        return os.path.splitext(p.name)[0]
    except Exception as e:
        return get_filename_without_extension(Path(p))


@return_result
def load_artifact(artifact_path):
    p = Path(artifact_path)
    if p.name.endswith("json"):
        return pd.read_json(p)
    else:
        with open(p, "rb") as f:
            artifact = pickle.load(f)
        return artifact


# +
pipeline_artifact_paths = [
    p for p in Path("../results").rglob("*")
    if p.name.endswith("json") or p.name.endswith("pkl")
]

pipeline_artifact_paths
# -

artifacts = {
    get_filename_without_extension(p): load_artifact(p)
    for p in pipeline_artifact_paths
}
artifacts.keys()

for k, v in artifacts.items():
    print(k, type(v))

generated_texts_df = artifacts["generated_texts_df"].unwrap()
generation_metrics_df = artifacts["generation_metrics_df"].unwrap()

evaluation_df = generation_metrics_df.groupby("repo").agg("mean").drop(columns=["generation", "index"])

s = generated_texts_df["tasks"].iloc[0][0]

import re

m = next(re.finditer("\[.*\]", s))

m.group()

# +
from tgutil.evaluation.preprocessing import EvalDFPreprocessor

preproc = EvalDFPreprocessor(id_col="", reference_text_col="", sanitize_re="\[(.*)\]")
# -

t = generation_metrics_df["generated_text"].iloc[0]

EvalDFPreprocessor.sanitize_str(t, "\[(.*)\]")

evaluation_df[["bleurt", "rouge1", "rouge2", "edit_word", "sentence_transformer_similarity"]].corr()

evaluation_df["bleurt"].plot.hist()

evaluation_df["rouge1"].plot.hist()

evaluation_df["sentence_transformer_similarity"].plot.hist()

import seaborn as sns

sns.heatmap(evaluation_df.corr(), annot=True)

sns.heatmap(evaluation_df[evaluation_df["rouge1"] > 0].corr(), annot=True)


