import logging
import os
from typing import Callable, Dict, List, Optional, Protocol, Set

import numpy as np
import sentence_transformers
import torch
from github_search.ir.evaluator_numba_helpers import *
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.util import cos_sim, dot_score
from torch import Tensor
from tqdm import trange

logger = logging.getLogger(__name__)


class Encoder(Protocol):
    """
    encoder that generalizes SentenceTransformers - Encoder[str]
    to encoders that can handle data with other types like graphs
    """

    def encode(
        self,
        inputs: List[str],
        show_progress_bar: bool = False,
        batch_size: int = 32,
        convert_to_tensor: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """encode raw data"""


class CustomInformationRetrievalEvaluatorImpl(SentenceEvaluator):
    """
    this class is based on InformationRetrievalEvaluator from sentence_transformers.
    Some functions for metrics calculation were reimplemented for performance.
    """

    def __init__(
        self,
        queries: Dict[str, str],  # qid => query
        corpus: Dict[str, str],  # cid => doc
        relevant_docs: Dict[str, Set[str]],  # qid => Set[cid]
        corpus_chunk_size: int = 50000,
        mrr_at_k: List[int] = [10],
        ndcg_at_k: List[int] = [10],
        accuracy_at_k: List[int] = [1, 3, 5, 10],
        precision_recall_at_k: List[int] = [1, 3, 5, 10],
        map_at_k: List[int] = [100],
        show_progress_bar: bool = False,
        batch_size: int = 32,
        name: str = "",
        write_csv: bool = True,
        score_functions: List[Callable[[Tensor, Tensor], Tensor]] = {
            "cos_sim": cos_sim,
            "dot_score": dot_score,
        },  # Score function, higher=more similar
        main_score_function: str = None,
    ):

        self.queries_ids = []
        for qid in queries:
            if qid in relevant_docs and len(relevant_docs[qid]) > 0:
                self.queries_ids.append(qid)

        self.queries = [queries[qid] for qid in self.queries_ids]

        self.corpus_ids = list(corpus.keys())
        self.corpus = [corpus[cid] for cid in self.corpus_ids]

        self.relevant_docs = relevant_docs
        self.corpus_chunk_size = corpus_chunk_size
        self.mrr_at_k = mrr_at_k
        self.ndcg_at_k = ndcg_at_k
        self.accuracy_at_k = accuracy_at_k
        self.precision_recall_at_k = precision_recall_at_k
        self.map_at_k = map_at_k

        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size
        self.name = name
        self.write_csv = write_csv
        self.score_functions = score_functions
        self.score_function_names = sorted(list(self.score_functions.keys()))
        self.main_score_function = main_score_function

        if name:
            name = "_" + name

        self.csv_file: str = "Information-Retrieval_evaluation" + name + "_results.csv"
        self.csv_headers = ["epoch", "steps"]

        for score_name in self.score_function_names:
            for k in accuracy_at_k:
                self.csv_headers.append("{}-Accuracy@{}".format(score_name, k))

            for k in precision_recall_at_k:
                self.csv_headers.append("{}-Precision@{}".format(score_name, k))
                self.csv_headers.append("{}-Recall@{}".format(score_name, k))

            for k in mrr_at_k:
                self.csv_headers.append("{}-MRR@{}".format(score_name, k))

            for k in ndcg_at_k:
                self.csv_headers.append("{}-NDCG@{}".format(score_name, k))

            for k in map_at_k:
                self.csv_headers.append("{}-MAP@{}".format(score_name, k))

    def __call__(
        self,
        encoder: Encoder,
        output_path: str = None,
        epoch: int = -1,
        steps: int = -1,
        *args,
        **kwargs
    ) -> float:
        if epoch != -1:
            out_txt = (
                " after epoch {}:".format(epoch)
                if steps == -1
                else " in epoch {} after {} steps:".format(epoch, steps)
            )
        else:
            out_txt = ":"

        logger.info(
            "Information Retrieval Evaluation on " + self.name + " dataset" + out_txt
        )

        scores = self.compute_metrices(model, *args, **kwargs)

        # Write results to disc
        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                fOut = open(csv_path, mode="w", encoding="utf-8")
                fOut.write(",".join(self.csv_headers))
                fOut.write("\n")

            else:
                fOut = open(csv_path, mode="a", encoding="utf-8")

            output_data = [epoch, steps]
            for name in self.score_function_names:
                for k in self.accuracy_at_k:
                    output_data.append(scores[name]["accuracy@k"][k])

                for k in self.precision_recall_at_k:
                    output_data.append(scores[name]["precision@k"][k])
                    output_data.append(scores[name]["recall@k"][k])

                for k in self.mrr_at_k:
                    output_data.append(scores[name]["mrr@k"][k])

                for k in self.ndcg_at_k:
                    output_data.append(scores[name]["ndcg@k"][k])

                for k in self.map_at_k:
                    output_data.append(scores[name]["map@k"][k])

            fOut.write(",".join(map(str, output_data)))
            fOut.write("\n")
            fOut.close()

        if self.main_score_function is None:
            return max(
                [
                    scores[name]["map@k"][max(self.map_at_k)]
                    for name in self.score_function_names
                ]
            )
        else:
            return scores[self.main_score_function]["map@k"][max(self.map_at_k)]

    def get_queries_result_lists(
        self,
        encoder: Encoder,
        corpus_encoder: Optional[Encoder] = None,
        corpus_embeddings: Tensor = None,
    ) -> Dict[str, float]:
        if corpus_encoder is None:
            corpus_encoder = encoder

        max_k = max(
            max(self.mrr_at_k),
            max(self.ndcg_at_k),
            max(self.accuracy_at_k),
            max(self.precision_recall_at_k),
            max(self.map_at_k),
        )

        # Compute embedding for the queries
        query_embeddings = encoder.encode(
            self.queries,
            show_progress_bar=self.show_progress_bar,
            batch_size=self.batch_size,
            convert_to_tensor=True,
        )

        queries_result_list = {}
        for name in self.score_functions:
            queries_result_list[name] = [[] for _ in range(len(query_embeddings))]

        # Iterate over chunks of the corpus
        for corpus_start_idx in trange(
            0,
            len(self.corpus),
            self.corpus_chunk_size,
            desc="Corpus Chunks",
            disable=not self.show_progress_bar,
        ):
            corpus_end_idx = min(
                corpus_start_idx + self.corpus_chunk_size, len(self.corpus)
            )

            # Encode chunk of corpus
            if corpus_embeddings is None:
                sub_corpus_embeddings = corpus_encoder.encode(
                    self.corpus[corpus_start_idx:corpus_end_idx],
                    show_progress_bar=False,
                    batch_size=self.batch_size,
                    convert_to_tensor=True,
                )
            else:
                sub_corpus_embeddings = corpus_embeddings[
                    corpus_start_idx:corpus_end_idx
                ]

            # Compute cosine similarites
            for name, score_function in self.score_functions.items():
                pair_scores = score_function(query_embeddings, sub_corpus_embeddings)

                # Get top-k values
                pair_scores_top_k_values, pair_scores_top_k_idx = torch.topk(
                    pair_scores,
                    min(max_k, len(pair_scores[0])),
                    dim=1,
                    largest=True,
                    sorted=False,
                )
                pair_scores_top_k_values = pair_scores_top_k_values.cpu().tolist()
                pair_scores_top_k_idx = pair_scores_top_k_idx.cpu().tolist()

                for query_itr in range(len(query_embeddings)):
                    for sub_corpus_id, score in zip(
                        pair_scores_top_k_idx[query_itr],
                        pair_scores_top_k_values[query_itr],
                    ):
                        corpus_id = self.corpus_ids[corpus_start_idx + sub_corpus_id]
                        queries_result_list[name][query_itr].append(
                            {"corpus_id": corpus_id, "score": score}
                        )
        return queries_result_list

    def compute_metrices(
        self,
        encoder: Encoder,
        corpus_encoder: Optional[Encoder] = None,
        corpus_embeddings: Tensor = None,
    ) -> Dict[str, float]:
        queries_result_list = self.get_queries_result_lists(
            encoder, corpus_encoder, corpus_embeddings
        )
        logger.info("Queries: {}".format(len(self.queries)))
        logger.info("Corpus: {}\n".format(len(self.corpus)))

        # Compute scores
        scores = {
            name: self.compute_metrics(queries_result_list[name])
            for name in self.score_functions
        }

        # Output
        for name in self.score_function_names:
            logger.info("Score-Function: {}".format(name))
            self.output_scores(scores[name])

        return scores

    def numba_accuracy(k_vals, top_hit_corpus_ids, query_relevant_docs):
        acc_counter = 0
        for k_val in k_vals:
            for hit_corpus_id in top_hits_corpus_ids:
                if hit in query_relevant_docs:
                    return 1
        return 0

    def compute_metrics(self, queries_result_list: List[object]):
        # Init score computation values
        num_hits_at_k = {k: 0 for k in self.accuracy_at_k}
        precisions_at_k = {k: [] for k in self.precision_recall_at_k}
        recall_at_k = {k: [] for k in self.precision_recall_at_k}
        MRR = {k: 0 for k in self.mrr_at_k}
        ndcg = {k: [] for k in self.ndcg_at_k}
        AveP_at_k = {k: [] for k in self.map_at_k}

        # Compute scores on results
        for query_itr in range(len(queries_result_list)):
            query_id = self.queries_ids[query_itr]

            # Sort scores
            top_hits = sorted(
                queries_result_list[query_itr], key=lambda x: x["score"], reverse=True
            )
            query_relevant_docs = self.relevant_docs[query_id]

            # Accuracy@k - We count the result correct, if at least one relevant doc is accross the top-k documents
            np_relevant_docs = np.array(list(query_relevant_docs))
            for k_val in self.accuracy_at_k:
                retrieved_corpus_ids = np.array(
                    [hit["corpus_id"] for hit in top_hits[0:k_val]]
                )
                # for hit in top_hits[0:k_val]:
                #    if hit["corpus_id"] in query_relevant_docs:
                #        num_hits_at_k[k_val] += 1
                #        break
                num_hits_at_k[k_val] += (
                    fast_count(retrieved_corpus_ids, np_relevant_docs) > 0
                )

                # Precision and Recall@k
                num_correct = fast_count(retrieved_corpus_ids, np_relevant_docs)

                precisions_at_k[k_val].append(num_correct / k_val)
                recall_at_k[k_val].append(num_correct / len(query_relevant_docs))

            # MRR@k
            for k_val in self.mrr_at_k:
                retrieved_corpus_ids = np.array(
                    [hit["corpus_id"] for hit in top_hits[0:k_val]]
                )
                mrr = compute_fast_mrr(retrieved_corpus_ids, np_relevant_docs).sum()
                MRR[k_val] += mrr

            # NDCG@k
            def compute_ndcgs(k_val):
                predicted_relevance = [
                    1 if top_hit["corpus_id"] in query_relevant_docs else 0
                    for top_hit in top_hits[0:k_val]
                ]
                true_relevances = [1] * len(query_relevant_docs)

                ndcg_value = self.compute_dcg_at_k(
                    predicted_relevance, k_val
                ) / self.compute_dcg_at_k(true_relevances, k_val)
                return ndcg_value

            for k_val in self.ndcg_at_k:
                ndcg[k_val].append(compute_ndcgs(k_val))

            def compute_avg_precision(k_val):
                num_correct = 0
                sum_precisions = 0

                for rank, hit in enumerate(top_hits[0:k_val]):
                    if hit["corpus_id"] in query_relevant_docs:
                        num_correct += 1
                        sum_precisions += num_correct / (rank + 1)

                avg_precision = sum_precisions / min(k_val, len(query_relevant_docs))
                return avg_precision

            # MAP@k
            for k_val in self.map_at_k:
                AveP_at_k[k_val].append(compute_avg_precision(k_val))

        # Compute averages
        for k in num_hits_at_k:
            num_hits_at_k[k] /= len(self.queries)

        for k in precisions_at_k:
            precisions_at_k[k] = np.mean(precisions_at_k[k])

        for k in recall_at_k:
            recall_at_k[k] = np.mean(recall_at_k[k])

        for k in ndcg:
            ndcg[k] = np.mean(ndcg[k])

        for k in MRR:
            MRR[k] /= len(self.queries)

        for k in AveP_at_k:
            AveP_at_k[k] = np.mean(AveP_at_k[k])

        return {
            "accuracy@k": num_hits_at_k,
            "precision@k": precisions_at_k,
            "recall@k": recall_at_k,
            "ndcg@k": ndcg,
            "mrr@k": MRR,
            "map@k": AveP_at_k,
        }

    def output_scores(self, scores):
        for k in scores["accuracy@k"]:
            logger.info("Accuracy@{}: {:.2f}%".format(k, scores["accuracy@k"][k] * 100))

        for k in scores["precision@k"]:
            logger.info(
                "Precision@{}: {:.2f}%".format(k, scores["precision@k"][k] * 100)
            )

        for k in scores["recall@k"]:
            logger.info("Recall@{}: {:.2f}%".format(k, scores["recall@k"][k] * 100))

        for k in scores["mrr@k"]:
            logger.info("MRR@{}: {:.4f}".format(k, scores["mrr@k"][k]))

        for k in scores["ndcg@k"]:
            logger.info("NDCG@{}: {:.4f}".format(k, scores["ndcg@k"][k]))

        for k in scores["map@k"]:
            logger.info("MAP@{}: {:.4f}".format(k, scores["map@k"][k]))

    @staticmethod
    def compute_dcg_at_k(relevances, k):
        dcg = 0
        for i in range(min(len(relevances), k)):
            dcg += relevances[i] / np.log2(i + 2)  # +2 as we start our idx at 0
        return dcg
