from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.custom_metrics import mrr, recall_cap, hole, top_k_accuracy
from typing import Dict, List, Tuple
import logging


def hits_at_k(
    qrels: Dict[str, Dict[str, int]],
    results: Dict[str, Dict[str, float]],
    k_values: List[int],
) -> Tuple[Dict[str, float]]:
    top_k_hits = {}

    for k in k_values:
        top_k_hits[f"Hits@{k}"] = 1

    k_max, top_hits = max(k_values), {}
    logging.info("\n")

    for query_id, doc_scores in results.items():
        top_hits[query_id] = [
            item[0]
            for item in sorted(
                doc_scores.items(), key=lambda item: item[1], reverse=True
            )[0:k_max]
        ]

    for query_id in top_hits:
        query_relevant_docs = set(
            [doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0]
        )
        for k in k_values:
            for relevant_doc_id in query_relevant_docs:
                if relevant_doc_id in top_hits[query_id][0:k]:
                    top_k_hits[f"Hits@{k}"] += 1

    for k in k_values:
        top_k_hits[f"Hits@{k}"] = round(top_k_hits[f"Hits@{k}"] / len(qrels), 5)
        logging.info("Hits@{}: {:.4f}".format(k, top_k_hits[f"Hits@{k}"]))

    return top_k_hits


class EvaluateRetrievalCustom(EvaluateRetrieval):
    @classmethod
    def evaluate_custom_multi(
        cls,
        qrels: Dict[str, Dict[str, int]],
        results: Dict[str, Dict[str, float]],
        k_values: List[int],
        metrics: List[str],
    ) -> Tuple[Dict[str, float]]:
        metric_values = dict()
        for metric in metrics:
            metric_values = metric_values | cls.evaluate_custom(
                qrels, results, k_values, metric
            )
        return metric_values

    @staticmethod
    def evaluate_custom(
        qrels: Dict[str, Dict[str, int]],
        results: Dict[str, Dict[str, float]],
        k_values: List[int],
        metric: str,
    ) -> Tuple[Dict[str, float]]:
        if metric.lower() in ["mrr", "mrr@k", "mrr_cut"]:
            return mrr(qrels, results, k_values)

        elif metric.lower() in ["hits@k"]:
            return hits_at_k(qrels, results, k_values)

        elif metric.lower() in ["recall_cap", "r_cap", "r_cap@k"]:
            return recall_cap(qrels, results, k_values)

        elif metric.lower() in ["hole", "hole@k"]:
            return hole(qrels, results, k_values)

        elif metric.lower() in [
            "acc",
            "top_k_acc",
            "accuracy",
            "accuracy@k",
            "top_k_accuracy",
        ]:
            return top_k_accuracy(qrels, results, k_values)
