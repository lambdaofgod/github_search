from typing import Any, List, Dict
import pandas as pd
import evaluate
from pydantic import BaseModel, Field

import torch
import sentence_transformers
import abc
from mlutil.feature_extraction.embeddings import load_gensim_embedding_model


class MetricsKwargs:
    kwargs = {
        "sentence_transformer_similarity": {
            "model_name": "paraphrase-distilroberta-base-v1"
        },
        "wmd": {"model_name": "glove-twitter-25"},
        "rouge": {"use_aggregator": False},
        "bleurt": {},
    }


class TextGenerationMetric(abc.ABC):
    @abc.abstractmethod
    def evaluate(self, texts_df, reference_text_col, predicted_text_col):
        pass

    @abc.abstractstaticmethod
    def load(metric_name):
        pass


class HuggingfaceMetric(TextGenerationMetric, BaseModel):
    name: str
    hf_metric: evaluate.EvaluationModule
    metric_kwargs: Dict[str, Any] = Field(default_factory=dict)

    def evaluate(
        self,
        texts_df: pd.DataFrame,
        reference_text_col: str,
        predicted_text_col: str,
    ):
        metric_kwargs = MetricsKwargs.kwargs.get(self.name, dict())
        reference_texts = texts_df[reference_text_col]
        predicted_texts = texts_df[predicted_text_col]
        scores = self.hf_metric.compute(
            predictions=predicted_texts.to_list(),
            references=reference_texts.to_list(),
            **metric_kwargs,
        )
        if len(scores.keys()) == 1:
            values = list(scores.values())[0]
            scores_pd = pd.Series(values, name=self.name)
        else:
            scores_pd = pd.DataFrame(scores)
        scores_pd.index = texts_df.index
        return scores_pd

    @staticmethod
    def load(metric_name, metric_kwargs):
        return HuggingfaceMetric(
            name=metric_name,
            hf_metric=evaluate.load(metric_name),
            metric_kwargs=metric_kwargs,
        )

    class Config:
        arbitrary_types_allowed = True


class WMDMetric(TextGenerationMetric, BaseModel):
    gensim_model: Any
    name: str = Field(default="wmd")

    def evaluate(self, texts_df, reference_text_col, predicted_text_col):
        wmdistances = []
        for pred_text, ref_text in zip(
            texts_df[reference_text_col], texts_df[predicted_text_col]
        ):
            distance = self.gensim_model.wmdistance(pred_text, ref_text)
            wmdistances.append(distance)
        return pd.DataFrame({"wmd": wmdistances}, index=texts_df.index)

    @staticmethod
    def load(model_name="fasttext-wiki-news-subwords-300"):
        model = load_gensim_embedding_model(model_name)
        return WMDMetric(gensim_model=model)

    class Config:
        arbitrary_types_allowed = True


class SentenceTransformersMetric(TextGenerationMetric, BaseModel):
    sentence_transformer: sentence_transformers.SentenceTransformer

    def evaluate(self, texts_df, reference_text_col, predicted_text_col):
        reference_texts = texts_df[reference_text_col].tolist()
        predicted_texts = texts_df[predicted_text_col].tolist()
        reference_embeddings = self.sentence_transformer.encode(
            reference_texts, convert_to_tensor=True
        )
        predicted_embeddings = self.sentence_transformer.encode(
            predicted_texts, convert_to_tensor=True
        )
        with torch.no_grad():
            similarities = (
                torch.nn.CosineSimilarity()(reference_embeddings, predicted_embeddings)
                .cpu()
                .numpy()
            )
        return pd.DataFrame(
            {"sentence_transformer_similarity": similarities}, index=texts_df.index
        )

    @staticmethod
    def load(model_name):
        return SentenceTransformersMetric(
            sentence_transformer=sentence_transformers.SentenceTransformer(model_name)
        )

    class Config:
        arbitrary_types_allowed = True


def load_metric(metric_name, model_name=None, **metric_kwargs):
    if metric_name == "wmd":
        return WMDMetric.load(model_name)
    elif metric_name == "sentence_transformer_similarity":
        return SentenceTransformersMetric.load(model_name)
    else:
        return HuggingfaceMetric.load(metric_name, metric_kwargs)
