from haystack.nodes import retriever as haystack_retriever
import pandas as pd


def get_result_metadata(retriever: haystack_retriever.BaseRetriever, query, topk=100):
    return [
        {**doc.meta, "score": doc.score}
        for doc in retriever.retrieve(query, top_k=topk)
    ]


def get_aggregated_results_df(
    retriever: haystack_retriever.BaseRetriever, query, topk, prefilter_topk=100
):
    query_results_df = pd.DataFrame(
        get_result_metadata(retriever, query, topk=prefilter_topk)
    )
    if len(query_results_df) == 0:
        return
    else:
        return (
            query_results_df.groupby("repo")
            .agg({"score": "mean", "tasks": "first"})
            .sort_values("score")
            .iloc[:topk]
        )


def evaluate_query_results(
    retriever: haystack_retriever.BaseRetriever, query, thresholds, topk=10
):
    df = get_aggregated_results_df(retriever, query, topk)
    if df is None:
        return {
            "accuracy": 0,
            **{f"accuracy@overlap={threshold}": 0 for threshold in thresholds},
            "precision": 0,
            **{f"precision@overlap={threshold}": 0 for threshold in thresholds},
        }
    return evaluate_query_results_df(query, df, thresholds)
