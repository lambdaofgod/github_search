from github_search.utils import load_config_yaml_key, pd_read_star
import pandas as pd
from github_search.ir.evaluator import (
    InformationRetrievalEvaluatorConfig,
    EmbedderPairConfig,
    InformationRetrievalColumnConfig,
    InformationRetrievalEvaluator,
)
import logging

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
    return pd.DataFrame(ir_results["cos_sim"])


def make_information_retrieval_evaluation_task():
    ir_config = InformationRetrievalEvaluatorConfig(**yaml.safe_load(f))
    pipe.add_function_step(
        name="evaluate_information_retrieval",
        function=evaluate_information_retrieval_step,
        function_kwargs=dict(
            searched_df="${expand_documents.generated_texts_df}", ir_config=ir_config
        ),
        function_return=["ir_results"],
        cache_executed_step=True,
    )
    return pipe


if __name__ == "__main__":
    results_df = evaluate_information_retrieval(
        search_df_path="../../output/nbow_data_test.parquet",
        evaluated_df_path="../../output/pipelines/rwkv-4-raven-7b_evaluated_records.json",
        column_config_type="dependencies_with_generated_tasks",
        embedder_config_type="mpnet",
    )

    print(results_df)
