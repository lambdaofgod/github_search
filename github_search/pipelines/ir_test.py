from information_retrieval_pipeline import (
    load_ir_config,
    load_ir_df,
)
from github_search.ir.evaluator import (
    InformationRetrievalEvaluator,
)

search_df_path = "../../output/nbow_data_test.parquet"
evaluated_df_path = (
    "../../output/pipelines/rwkv-4-raven-7b_evaluated_records_micro.json"
)


ir_df = load_ir_df(search_df_path, evaluated_df_path)
ir_config = load_ir_config(search_df_path, "dependencies", "dependencies_best_model")
ir_evaluator = InformationRetrievalEvaluator.setup_from_df(ir_df, ir_config)
ir_results = ir_evaluator.evaluate()

ir_results.aggregate_metrics.to_csv("/tmp/ir_results_agg.csv")
ir_results.per_query_metrics.to_csv("/tmp/ir_results_query.csv")
