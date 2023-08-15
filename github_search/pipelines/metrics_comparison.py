import ast
import numpy as np
import tqdm
from typing import Any
from github_search.utils import load_config_yaml_key, pd_read_star
import pandas as pd
from github_search.ir.evaluator import (
    InformationRetrievalEvaluatorConfig,
    EmbedderPairConfig,
    InformationRetrievalColumnConfig,
    InformationRetrievalEvaluator,
)
from github_search.ir.models import InformationRetrievalMetricsResult
import logging
from clearml import Task, Logger
import itertools
import yaml
import pandera as pa
from typing import List, Dict, Optional
from pydantic import BaseModel
from typing import List, Dict, Optional, Tuple

logging.basicConfig(level=logging.INFO)


class RunConfig(BaseModel, frozen=True):
    column_config_type: str
    embedder_config_type: str
    ir_config: InformationRetrievalEvaluatorConfig

    @classmethod
    def load(cls, column_config_type, embedder_config_type):
        ir_config = load_ir_config(column_config_type, embedder_config_type)
        return RunConfig(
            column_config_type=column_config_type,
            embedder_config_type=embedder_config_type,
            ir_config=ir_config,
        )


class MetricsDFs(BaseModel):
    search_df: pd.DataFrame
    generation_eval_df: pd.DataFrame

    class Config:
        arbitrary_types_allowed = True


def replace_list_chars(text):
    return text.replace("[", "").replace("]", "").replace(",", "").replace("'", "")


def process_generated_text(text):
    return replace_list_chars(text.strip().split("\n")[0])


def make_search_df(input_search_df, evaluated_df):
    max_len = 100
    search_df = input_search_df[["repo", "dependencies"]].merge(evaluated_df, on="repo")
    search_df = search_df.assign(
        generated_tasks=search_df["generated_text"].apply(process_generated_text)
    )
    search_df = search_df.assign(
        truncated_dependencies=search_df["dependencies"].apply(
            lambda doc: " ".join(doc.split()[:max_len])
        )
    )
    return search_df


def load_ir_config(column_config_type, embedder_config_type):
    logging.info("Loading configs")
    embedder_config = load_config_yaml_key(
        EmbedderPairConfig, "conf/retrieval.yaml", embedder_config_type
    )
    column_config = load_config_yaml_key(
        InformationRetrievalColumnConfig, "conf/column_configs.yaml", column_config_type
    )
    return InformationRetrievalEvaluatorConfig(
        embedder_config=embedder_config,
        column_config=column_config,
    )


def evaluate_information_retrieval(input_search_df, evaluated_df, ir_config):
    search_df = make_search_df(input_search_df, evaluated_df)
    ir_evaluator = InformationRetrievalEvaluator.setup_from_df(search_df, ir_config)
    logging.info("Loaded evaluator, evaluating...")
    return ir_evaluator.evaluate()


class MetricComparator:
    @pa.check_input(
        pa.DataFrameSchema(
            {
                "true_tasks": pa.Column(List[str]),
                "generated_text": pa.Column(str),
                "repo": pa.Column(str),
            }
        ),
        "evaluated_df",
    )
    def get_reported_metrics(
        self, ir_results: InformationRetrievalMetricsResult, evaluated_df: pd.DataFrame
    ):
        evaluated_df = evaluated_df.rename(columns={"tasks": "task"}).explode("task")
        query_metrics_df = (
            ir_results.per_query_metrics.merge(
                evaluated_df, left_index=True, right_on="task"
            )
            .groupby("task")
            .mean()
        )
        ir_metric_names = ir_results.per_query_metrics.columns
        generation_metric_names = evaluated_df.select_dtypes(include="number").columns
        ir_vs_generation_metrics_df = query_metrics_df.corr(method="kendall").loc[
            ir_metric_names, generation_metric_names
        ]
        return {
            "ir_metrics_df": pd.DataFrame(ir_results.aggregate_metrics),
            "ir_vs_generation_metrics_df": ir_vs_generation_metrics_df,
            "mean_generation_metrics": evaluated_df.mean().to_dict(),
            "mean_ir_metrics": ir_results.aggregate_metrics.loc["mean"].to_dict(),
            "generation_metrics_summary": evaluated_df.describe(),
        }

    def _report_iteration_metrics(self, logger, reported_metrics, iteration):
        logger.report_table(
            title="metrics",
            series="ir_metrics_aggregated",
            table_plot=reported_metrics["ir_metrics_df"],
            iteration=iteration,
        )
        logger.report_table(
            title="generation_metrics",
            series="generation_metrics",
            table_plot=reported_metrics["generation_metrics_summary"],
            iteration=iteration,
        )
        logger.report_table(
            title="ir_vs_generation_metrics",
            series="ir_vs_generation_metrics",
            table_plot=reported_metrics["ir_vs_generation_metrics_df"],
            iteration=iteration,
        )
        for metrics_dict in [
            reported_metrics["mean_generation_metrics"],
            reported_metrics["mean_ir_metrics"],
        ]:
            for metric_name, value in metrics_dict.items():
                title = metric_name
                logger.report_scalar(
                    title=title, series=title, value=value, iteration=iteration
                )

    def report_metrics(self, logger, ir_results, evaluated_df, iteration):
        reported_metrics = self.get_reported_metrics(ir_results, evaluated_df)
        self._report_iteration_metrics(logger, reported_metrics, iteration)

    def run(
        self, run_config: RunConfig, metrics_dfs
    ) -> Dict[int, Tuple[InformationRetrievalMetricsResult, pd.DataFrame]]:
        return dict(
            self.run_evaluation(
                metrics_dfs.generation_eval_df,
                metrics_dfs.search_df,
                run_config.ir_config,
            )
        )

    @pa.check_input(
        pa.DataFrameSchema(
            {
                "true_tasks": pa.Column(List[str]),
                "generated_text": pa.Column(str),
                "repo": pa.Column(str),
                "generation": pa.Column(int),
            }
        ),
        "evaluated_df",
    )
    def run_evaluation(
        self, evaluated_df, input_search_df, ir_config
    ) -> Dict[int, Tuple[InformationRetrievalMetricsResult, pd.DataFrame]]:
        """
        run information retrieval evaluation
        for each generation from evalueated_df
        """
        for generation_id in set(evaluated_df["generation"]):
            chunk_evaluated_df = evaluated_df[
                evaluated_df["generation"] == generation_id
            ]
            ir_result = evaluate_information_retrieval(
                input_search_df, chunk_evaluated_df, ir_config
            )
            yield generation_id, (
                ir_result,
                chunk_evaluated_df,
            )


class EmbedderConfig(BaseModel):
    doc_max_length: Optional[int]
    document_embedder_path: str
    query_embedder_path: str
    query_max_length: Optional[int]


class MetricComparisonConfig(BaseModel):
    embedder_configs: Dict[str, EmbedderConfig]
    column_configs: Dict[str, dict]

    @classmethod
    def load(cls, embedder_config_path, column_config_path):
        with open(embedder_config_path) as f:
            embedder_configs = {
                k: EmbedderConfig(**conf) for k, conf in yaml.safe_load(f).items()
            }
        with open(column_config_path) as f:
            column_configs = {k: conf for k, conf in yaml.safe_load(f).items()}

        return MetricComparisonConfig(
            embedder_configs=embedder_configs,
            column_configs=column_configs,
        )


DataFrameRecords = List[Dict[str, Any]]


class MetricsExperimentResult(BaseModel):
    """
    this class is what is stored per run in ZenML

    we have to write some boilerplate because
    ZenML can't guess that dict with pandas dataframe is serializable
    """

    ir_config: InformationRetrievalEvaluatorConfig
    per_query_metrics: Dict[int, DataFrameRecords]
    aggregate_metrics: Dict[int, DataFrameRecords]
    generation_metrics: Dict[int, DataFrameRecords]

    @classmethod
    def create_from_results(
        cls,
        ir_config,
        run_results: Dict[int, Tuple[InformationRetrievalMetricsResult, pd.DataFrame]],
    ):
        per_query_metrics = {
            generation_id: result.per_query_metrics.to_dict(orient="records")
            for (generation_id, (result, _)) in run_results.items()
        }
        aggregate_metrics = {
            generation_id: result.per_query_metrics.to_dict(orient="records")
            for (generation_id, (result, _)) in run_results.items()
        }
        generation_metrics = {
            generation_id: cls.drop_invalid_dtypes(df).to_dict(orient="records")
            for (generation_id, (_, df)) in run_results.items()
        }
        return MetricsExperimentResult(
            ir_config=ir_config,
            per_query_metrics=per_query_metrics,
            aggregate_metrics=aggregate_metrics,
            generation_metrics=generation_metrics,
        )

    @classmethod
    def drop_invalid_dtypes(cls, df):
        dtypes = {col: type(df[col].iloc[0]) for col in df.columns}
        return df[[col for (col, dtype) in dtypes.items() if not dtype is np.ndarray]]

    class Config:
        arbitrary_types_allowed = True
        frozen = True


def get_run_metrics(
    config: MetricComparisonConfig,
    search_df: pd.DataFrame,
    generation_eval_df: pd.DataFrame,
):
    configs_grid = list(
        itertools.product(config.column_configs.keys(), config.embedder_configs.keys())
    )

    generation_eval_df = generation_eval_df.assign(
        true_tasks=generation_eval_df["true_tasks"].apply(list)
    )

    metrics_dfs = MetricsDFs(search_df=search_df, generation_eval_df=generation_eval_df)

    for column_config_type, embedder_type in tqdm.tqdm(configs_grid):
        run_config = RunConfig.load(
            column_config_type,
            embedder_type,
        )
        run_results = MetricComparator().run(run_config, metrics_dfs)

        yield MetricsExperimentResult.create_from_results(
            ir_config=run_config.ir_config,
            run_results=run_results,
        )


if __name__ == "__main__":
    conf_dict = dict(
        embedder_config_path="conf/retrieval.yaml",
        column_config_path="conf/column_configs.yaml",
    )
    config = MetricComparisonConfig.load(**conf_dict)
    list(compare_metrics(config))
