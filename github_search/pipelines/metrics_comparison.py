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

logging.basicConfig(level=logging.INFO)


def replace_list_chars(text):
    return text.replace("[", "").replace("]", "").replace(",", "").replace("'", "")


def process_generated_text(text):
    return replace_list_chars(text.strip().split("\n")[0])


def make_ir_df(input_ir_df, evaluated_df):
    max_len = 100
    ir_df = input_ir_df[["repo", "dependencies"]].merge(evaluated_df, on="repo")
    ir_df = ir_df.assign(
        generated_tasks=ir_df["generated_text"].apply(process_generated_text)
    )
    ir_df = ir_df.assign(
        truncated_dependencies=ir_df["dependencies"].apply(
            lambda doc: " ".join(doc.split()[:max_len])
        )
    )
    return ir_df


def load_ir_df(search_df_path, evaluated_df_path):
    return make_ir_df(input_ir_df, evaluated_df)


def load_ir_config(search_df_path, column_config_type, embedder_config_type):
    logging.info("Loading configs")
    embedder_config = load_config_yaml_key(
        EmbedderPairConfig, "conf/retrieval.yaml", embedder_config_type
    )
    column_config = load_config_yaml_key(
        InformationRetrievalColumnConfig, "conf/column_configs.yaml", column_config_type
    )
    return InformationRetrievalEvaluatorConfig(
        search_df_path=search_df_path,
        embedder_config=embedder_config,
        column_config=column_config,
    )


def evaluate_information_retrieval(input_ir_df, evaluated_df, ir_config):
    ir_df = make_ir_df(input_ir_df, evaluated_df)
    ir_evaluator = InformationRetrievalEvaluator.setup_from_df(ir_df, ir_config)
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
        self,
        search_df_path,
        evaluated_df_path,
        column_config_type,
        embedder_config_type,
    ):
        task = Task.create(
            project_name="github_search",
            task_name=f"information_retrieval_evaluation_{embedder_config_type}_{column_config_type}",
        )
        ir_config = load_ir_config(
            search_df_path, column_config_type, embedder_config_type
        )
        params = ir_config.dict()
        params["search_df_path"] = search_df_path
        params["evaluated_df_path"] = evaluated_df_path
        logging.info(f"running task with {params}")
        task.connect(params)
        input_ir_df = pd_read_star(search_df_path)
        evaluated_df = pd.read_json(evaluated_df_path, orient="records", lines=True)
        logger = task.get_logger()
        for generation_id in set(evaluated_df["generation_id"]):
            chunk_evaluated_df = evaluated_df[
                evaluated_df["generation_id"] == generation_id
            ]
            ir_result = evaluate_information_retrieval(
                input_ir_df, chunk_evaluated_df, ir_config
            )
            self.report_metrics(logger, ir_result, evaluated_df, generation_id)
            logger.flush()


class EmbedderConfig(BaseModel):
    doc_max_length: Optional[int]
    document_embedder_path: str
    query_embedder_path: str
    query_max_length: Optional[int]
    search_df_path: str


class MetricComparisonConfig(BaseModel):
    embedder_configs: Dict[str, EmbedderConfig]
    column_configs: Dict[str, dict]
    search_df_path: str
    evaluated_df_path: str

    @classmethod
    def load(
        embedder_config_path, column_config_path, search_df_path, evaluated_df_path
    ):
        with open(embedder_config_path) as f:
            embedder_configs = {
                k: EmbedderConfig(**conf) for k, conf in yaml.safe_load(f)
            }
        with open(column_config_path) as f:
            column_configs = {k: conf for k, conf in yaml.safe_load(f)}

        return MetricComparisonConfig(
            embedder_configs=embedder_configs, column_configs=column_configs
        )


if __name__ == "__main__":
    conf_dict = dict(
        embedder_config_path="conf/retrieval.yaml",
        column_config_path="conf/column_configs.yaml",
        search_df_path="../../output/nbow_data_test.parquet",
        evaluated_df_path=(
            "../../output/pipelines/api_rwkv_evaluated_records_no_sampling.json"
        ),
    )

    config = MetricComparisonConfig.load(**conf_dict)

    configs_grid = list(
        itertools.product(config.column_configs.keys(), config.embedder_configs.keys())
    )

    for column_config_type, embedder_type in configs_grid:
        MetricComparator().run(
            search_df_path=config.search_df_path,
            evaluated_df_path=config.evaluated_df_path,
            embedder_config_type=embedder_type,
            column_config_type=column_config_type,
        )
