from github_search.utils import load_config_yaml_key, pd_read_star
import pandas as pd
from github_search.ir.evaluator import (
    InformationRetrievalEvaluatorConfig,
    EmbedderPairConfig,
    InformationRetrievalColumnConfig,
    InformationRetrievalEvaluator,
)
import logging
from clearml import Task, Logger
import itertools
import yaml

logging.basicConfig(level=logging.INFO)


def replace_list_chars(text):
    return text.replace("[", "").replace("]", "").replace(",", "").replace("'", "")


def process_generated_text(text):
    return replace_list_chars(text.strip().split("\n")[0])


def make_ir_df(ir_corpus_df, evaluated_df):
    max_len = 100
    ir_df = ir_corpus_df[["repo", "dependencies"]].merge(evaluated_df, on="repo")
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
    ir_corpus_df = pd_read_star(search_df_path)
    evaluated_df = pd.read_json(evaluated_df_path, orient="records", lines=True)
    return make_ir_df(ir_corpus_df, evaluated_df)


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


def evaluate_information_retrieval(search_df_path, evaluated_df_path, ir_config):
    logging.info(f"loaded ir config:\n {str(dict(ir_config))}")
    logging.info("Loading information retrieval df")
    ir_df = load_ir_df(search_df_path, evaluated_df_path)
    ir_evaluator = InformationRetrievalEvaluator.setup_from_df(ir_df, ir_config)
    logging.info("Loaded evaluator, evaluating...")
    return ir_evaluator.evaluate()


def report_ir_metrics(logger, ir_results):
    logger.report_table(
        title="metrics",
        series="ir_metrics_aggregated",
        table_plot=pd.DataFrame(ir_results.aggregate_metrics),
        iteration=0,
    )
    mean_metrics_dict = ir_results.aggregate_metrics.loc["mean"].to_dict()
    for metric_name, value in mean_metrics_dict.items():
        title = metric_name
        logger.report_scalar(title=title, series=title, value=value, iteration=0)


def report_generation_metrics(logger, evaluated_df):
    metrics = evaluated_df.mean().to_dict()
    for metric_name, value in metrics.items():
        logger.report_scalar(
            title=metric_name, series=metric_name, value=value, iteration=0
        )

    logger.report_table(
        title="generation_metrics",
        series="generation_metrics",
        table_plot=evaluated_df.describe(),
        iteration=0,
    )


def report_metrics(logger, ir_results, evaluated_df):
    report_ir_metrics(logger, ir_results)
    report_generation_metrics(logger, evaluated_df)
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
    ir_generation_metrics_df = query_metrics_df.corr(method="kendall").loc[
        ir_metric_names, generation_metric_names
    ]
    logger.report_table(
        title="query_metrics",
        series="query_metrics",
        table_plot=pd.DataFrame(query_metrics_df),
        iteration=0,
    )
    logger.report_table(
        title="metrics",
        series="ir_vs_generation_metrics",
        table_plot=pd.DataFrame(ir_generation_metrics_df),
        iteration=0,
    )


def run_information_retrieval_evaluation_task(
    search_df_path, evaluated_df_path, column_config_type, embedder_config_type
):
    task = Task.create(
        project_name="github_search",
        task_name=f"information_retrieval_evaluation_{embedder_config_type}_{column_config_type}",
    )
    ir_config = load_ir_config(search_df_path, column_config_type, embedder_config_type)
    params = ir_config.dict()
    params["search_df_path"] = search_df_path
    params["evaluated_df_path"] = evaluated_df_path
    logging.info(f"running task with {params}")
    task.connect(params)
    evaluated_df = pd.read_json(evaluated_df_path, orient="records", lines=True)
    ir_result = evaluate_information_retrieval(
        search_df_path, evaluated_df_path, ir_config
    )
    logger = task.get_logger()
    report_metrics(logger, ir_result, evaluated_df)
    logger.flush()


if __name__ == "__main__":
    with open("conf/retrieval.yaml") as f:
        embedder_types = yaml.safe_load(f).keys()
    with open("conf/column_configs.yaml") as f:
        column_config_types = yaml.safe_load(f).keys()

    search_df_path = "../../output/nbow_data_test.parquet"
    evaluated_df_path = (
        "../../output/pipelines/api_rwkv_evaluated_records_no_sampling.json"
    )

    configs_grid = list(itertools.product(column_config_types, embedder_types))
    for (column_config_type, embedder_type) in configs_grid:
        run_information_retrieval_evaluation_task(
            search_df_path=search_df_path,
            evaluated_df_path=evaluated_df_path,
            embedder_config_type=embedder_type,
            column_config_type=column_config_type,
        )
