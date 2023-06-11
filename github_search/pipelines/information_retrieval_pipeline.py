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


def evaluate_information_retrieval(
    search_df_path, evaluated_df_path, column_config_type, embedder_config_type
):
    logging.info("Loading configs")
    embedder_config = load_config_yaml_key(
        EmbedderPairConfig, "conf/retrieval.yaml", embedder_config_type
    )
    column_config = load_config_yaml_key(
        InformationRetrievalColumnConfig, "conf/column_configs.yaml", column_config_type
    )
    ir_config = InformationRetrievalEvaluatorConfig(
        search_df_path=search_df_path,
        embedder_config=embedder_config,
        column_config=column_config,
    )
    logging.info(f"loaded ir config:\n {str(dict(ir_config))}")
    logging.info("Loading information retrieval df")
    ir_df = load_ir_df(search_df_path, evaluated_df_path)
    ir_evaluator = InformationRetrievalEvaluator.setup_from_df(ir_df, ir_config)
    logging.info("Loaded evaluator, evaluating...")
    ir_results = ir_evaluator.evaluate()
    return ir_results["cos_sim"]


def report_ir_metrics(logger, ir_results):
    logger.report_table(
        title="ir_metrics",
        series="ir_metrics",
        matrix=pd.DataFrame(ir_results),
        iteration=0,
    )
    for metric_name, k_by_metric in ir_results.items():
        for k, v in k_by_metric.items():
            title = f"{metric_name}@{k}"
            logger.report_scalar(title=title, series=title, value=v, iteration=0)


def report_generation_metrics(logger, evaluated_df):
    metrics = evaluated_df.mean().to_dict()
    for metric_name, value in metrics.items():
        logger.report_scalar(
            title=metric_name, series=metric_name, value=value, iteration=0
        )

    logger.report_table(
        title="generation_metrics",
        series="generation_metrics",
        matrix=evaluated_df.describe(),
        iteration=0,
    )


def run_information_retrieval_evaluation_task(
    search_df_path, evaluated_df_path, column_config_type, embedder_config_type
):
    task = Task.create(
        project_name="github_search", task_name="information_retrieval_evaluation"
    )
    params = dict(
        search_df_path=search_df_path,
        evaluated_df_path=evaluated_df_path,
        column_config_type=column_config_type,
        embedder_config_type=embedder_config_type,
    )
    task.connect(params)
    evaluated_df = pd.read_json(evaluated_df_path, orient="records", lines=True)
    ir_result = evaluate_information_retrieval(
        search_df_path, evaluated_df_path, column_config_type, embedder_config_type
    )
    logger = task.get_logger()
    report_ir_metrics(logger, ir_result)
    report_generation_metrics(logger, evaluated_df)
    logger.flush()


if __name__ == "__main__":
    __ = run_information_retrieval_evaluation_task(
        search_df_path="../../output/nbow_data_test.parquet",
        evaluated_df_path="../../output/pipelines/rwkv-4-raven-7b_evaluated_records.json",
        column_config_type="dependencies_with_generated_tasks",
        embedder_config_type="mpnet",
    )
