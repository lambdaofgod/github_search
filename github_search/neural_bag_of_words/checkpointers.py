import os
from typing import List

from toolz import partial
import yaml
from github_search.ir import evaluator
from github_search.neural_bag_of_words import embedders


class EncoderCheckpointer:
    def __init__(
        self,
        dset,
        val_df,
        nbow_query,
        nbow_document,
        save_dir,
        epochs: List[int],
        eval_max_seq_length: int = 5000,
        save_models=True,
    ):
        self.save_models = save_models
        self.get_query_embedder = partial(
            embedders.make_sentence_transformer_nbow_model,
            vocab=dset.query_numericalizer.vocab.vocab.itos_,
            tokenize_fn=dset.query_numericalizer.tokenizer,
            token_weights=nbow_query.token_weights.cpu().numpy().tolist(),
            max_seq_length=100,
        )

        self.get_document_embedder = partial(
            embedders.make_sentence_transformer_nbow_model,
            vocab=dset.document_numericalizer.vocab.vocab.itos_,
            tokenize_fn=dset.document_numericalizer.tokenizer,
            token_weights=nbow_document.token_weights.cpu().numpy().tolist(),
            max_seq_length=eval_max_seq_length,
        )
        self.epochs = epochs + ["final"]
        self.val_df = val_df
        self.save_dir = save_dir

    def get_ir_evaluator(self, query_embedder, document_embedder):
        ir_evaluator = evaluator.InformationRetrievalEvaluator(
            query_embedder, document_embedder
        )
        ir_evaluator.setup(self.val_df.reset_index(), "tasks", "dependencies")
        return ir_evaluator

    def save_epoch_checkpoint(self, nbow_query, nbow_document, epoch):
        if epoch in self.epochs:
            query_embedder = self.get_query_embedder(nbow_query)
            document_embedder = self.get_document_embedder(nbow_document)
            embedders = (query_embedder, document_embedder)
            metrics = self.get_epoch_metrics(*embedders)
            self.unconditionally_save_epoch_checkpoint(
                epoch, *embedders, metrics=metrics
            )

    def get_epoch_metrics(self, query_embedder, document_embedder):
        ir_evaluator = self.get_ir_evaluator(query_embedder, document_embedder)
        return ir_evaluator.evaluate()

    def unconditionally_save_epoch_checkpoint(
        self, epoch, query_embedder, document_embedder, metrics
    ):
        if self.save_models:
            query_embedder.save(os.path.join(self.save_dir, f"nbow_query_{epoch}"))
            document_embedder.save(
                os.path.join(self.save_dir, f"nbow_document_{epoch}")
            )

        metrics_path = os.path.join(self.save_dir, f"metrics_{epoch}.yaml")
        print(yaml.dump(metrics["cos_sim"]))
        with open(metrics_path, "w") as f:
            yaml.dump(metrics, f)
