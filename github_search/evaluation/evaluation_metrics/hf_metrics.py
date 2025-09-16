from typing import Any, Dict
import pandas as pd
import evaluate
from pydantic import BaseModel, Field

from tgutil.evaluation_metrics.base import TextGenerationMetric
from tgutil.type_utils import StrEnum


class HuggingfaceMetricName(StrEnum):
    rouge = "rouge"
    bertscore = "bertscore"


class HuggingfaceMetric(TextGenerationMetric, BaseModel):
    name: HuggingfaceMetricName
    hf_metric: evaluate.EvaluationModule
    metric_kwargs: Dict[str, Any] = Field(default_factory=dict)

    def evaluate(
        self,
        texts_df: pd.DataFrame,
        reference_text_col: str,
        predicted_text_col: str,
    ):
        reference_texts = texts_df[reference_text_col]
        predicted_texts = texts_df[predicted_text_col]
        scores = self.hf_metric.compute(
            predictions=predicted_texts.to_list(),
            references=reference_texts.to_list(),
            **self.metric_kwargs,
        )
        if len(scores.keys()) == 1:
            values = list(scores.values())[0]
            scores_pd = pd.Series(values, name=self.name)
        else:
            scores_pd = pd.DataFrame(scores)
        scores_pd.index = texts_df.index
        return scores_pd

    @classmethod
    def load(cls, metric_name, metric_kwargs):
        metric_init_args = cls.get_kwargs(metric_name, "metric_init_args")
        metric_kwargs = cls.get_kwargs(metric_name, "metric_kwargs")
        return HuggingfaceMetric(
            name=metric_name,
            hf_metric=evaluate.load(metric_name, *metric_init_args),
            metric_kwargs=metric_kwargs,
        )

    @classmethod
    def get_kwargs(cls, name, kwargs_type):
        if kwargs_type == "metric_kwargs":
            return cls.Config.kwargs.get(name, dict())
        elif kwargs_type == "metric_init_args":
            return cls.Config.args.get(name, [])

    class Config:
        arbitrary_types_allowed = True
        kwargs = {
            "rouge": {"use_aggregator": False},
            "bertscore": {"lang": "en"},
        }
        args = {}
