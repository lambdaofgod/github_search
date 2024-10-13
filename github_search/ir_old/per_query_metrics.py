from typing import Dict, List, Tuple


def get_hits_at_k(
    qrels: Dict[str, Dict[str, int]],
    results: Dict[str, Dict[str, float]],
    k_values: List[int],
) -> Tuple[Dict[str, float]]:
    per_query_hits = {}

    k_max, top_hits = max(k_values), {}

    for query_id, doc_scores in results.items():
        top_hits[query_id] = [
            item[0]
            for item in sorted(
                doc_scores.items(), key=lambda item: item[1], reverse=True
            )[0:k_max]
        ]

    for query_id in top_hits:
        hits = {}
        for k in k_values:
            hits[f"Accuracy@{k}"] = 0.0
        query_relevant_docs = set(
            [doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0]
        )
        for k in k_values:
            for relevant_doc_id in query_relevant_docs:
                if relevant_doc_id in top_hits[query_id][0:k]:
                    hits[f"Accuracy@{k}"] += 1.0
        per_query_hits[query_id] = hits

    return per_query_hits
