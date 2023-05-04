import pandas as pd
from operator import itemgetter
import re
from evaluation_metrics import TextGenerationMetric, load_metric, MetricsKwargs
from pydantic import BaseModel
from github_search.utils import load_paperswithcode_df
from typing import List, Dict, Union

import pandas as pd
from operator import itemgetter


class EvalDFPreprocessor:
    @classmethod
    def get_eval_df_from_raw_generated_text(
        cls, generated_text_df: pd.DataFrame, repo_tasks_df: pd.DataFrame
    ):
        """
        generated_text_df is assumed to have the following columns:
        - generated_text
        """
        # task_list_df = generated_text_df["predicted_text"].apply(get_task_list)
        repo_tasks_df = repo_tasks_df.rename({"tasks": "true_tasks"}, axis=1)[
            ["repo", "true_tasks"]
        ]
        generated_text_df["generated_text"] = generated_text_df["generated_text"].apply(
            cls.preprocess_raw_generated_text
        )
        evaluation_df = cls.prepare_texts_for_scoring(
            generated_text_df.merge(repo_tasks_df, on="repo")
        )
        return evaluation_df

    @classmethod
    def get_eval_dffrom_generated_text(cls, generated_text_df: pd.DataFrame):
        """
        generated_text_df is assumed to have the following columns:
        - generated_text
        """
        return cls.prepare_texts_for_scoring(generated_text_df)

    @classmethod
    def prepare_texts_for_scoring(cls, texts_df):
        pred_raw_texts = (
            texts_df["generated_text"]
            .str.strip()
            .str.split("\n")
            .apply(itemgetter(0))
            .apply(cls.sanitize_list_str)
        )
        texts_df["predicted_text"] = pred_raw_texts.apply(
            EvalDFPreprocessor.sanitize_list_str
        )
        texts_df["reference_text"] = texts_df["true_tasks"].str.join(", ")
        return texts_df

    @classmethod
    def preprocess_raw_generated_text(cls, out_text):
        return out_text.split("## tags")[-1].strip()

    @classmethod
    def get_task_list(cls, task_line):
        return task_line.split(", ")

    @classmethod
    def sanitize_list_str(cls, s):
        return re.sub(r'[\[\]\'"]', "", s)

    @classmethod
    def postprocess_reference_texts(cls, texts_df, reference_text_col):
        texts_df["true_tasks"].str.join(", ")


class TextGenerationEvaluator(BaseModel):
    evaluation_metrics: List[TextGenerationMetric]

    @classmethod
    def from_metric_names(
        cls,
        metric_names=["bleurt", "rouge", "wmd", "sentence_transformer_similarity"],
        metric_kwargs=MetricsKwargs.kwargs,
    ):
        metrics = [
            load_metric(metric_name, **metric_kwargs[metric_name])
            for metric_name in metric_names
        ]
        return cls(evaluation_metrics=metrics)

    def get_scores(
        self,
        texts_df,
    ):
        generation_metrics_dfs = [
            metric.evaluate(texts_df, "reference_text", "predicted_text")
            for metric in self.evaluation_metrics
        ]
        return pd.concat(generation_metrics_dfs, axis=1)

    def get_evaluated_df(self, texts_df):
        return pd.concat(
            [texts_df, self.get_scores(texts_df)],
            axis=1,
        )

    class Config:
        arbitrary_types_allowed = True


if __name__ == "__main__":

    generated_tasks_df = pd.read_json(
        "outputs/llama-13b-hf_generated.jsonl", orient="records", lines=True
    )
    repo_tasks_df = load_paperswithcode_df("../../data/paperswithcode_with_tasks.csv")
    texts_df = EvalDFPreprocessor.get_eval_df_from_raw_generated_text(
        generated_tasks_df, repo_tasks_df
    )
    eval_df = (
        TextGenerationEvaluator.from_metric_names(
            metric_names=["bleurt", "rouge", "wmd", "sentence_transformer_similarity"]
        )
        .get_evaluated_df(texts_df=texts_df)
        .sort_values(by="rougeL", ascending=False)
    )
    eval_df.to_json(
        "outputs/llama-13b-hf_generated_eval.json", orient="records", lines=True
    )
