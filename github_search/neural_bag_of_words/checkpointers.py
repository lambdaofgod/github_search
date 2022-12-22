import os
from typing import List

from torch import nn
from toolz import partial
import yaml
from github_search.ir import evaluator, InformationRetrievalColumnConfig, models
from github_search.neural_bag_of_words import embedders, evaluation, training_utils
from dataclasses import dataclass
import sentence_transformers


class EncoderCheckpointer:
    def __init__(
        self,
        dset,
        val_df,
        embedder_pair: models.EmbedderPair,
        save_dir: str,
        column_config: InformationRetrievalColumnConfig,
        epochs: List[int],
        eval_max_seq_length: int = 5000,
        save_models=True,
    ):
        self.column_config = column_config
        self.dset = dset
        self.save_models = save_models
        self.embedder_pair = embedder_pair
        self.epochs = epochs + ["final"]
        self.val_df = val_df.copy()
        self.save_dir = save_dir

    def get_ir_evaluator(self, embedder_pair: evaluator.EmbedderPair):
        ir_evaluator = evaluator.InformationRetrievalEvaluator(embedder_pair)
        ir_evaluator.setup(self.val_df, self.column_config)
        return ir_evaluator

    def save_epoch_checkpoint(self, embedder_pair, epoch):
        if epoch in self.epochs:
            self.unconditionally_save_epoch_checkpoint(
                epoch=epoch, embedder_pair=embedder_pair
            )

    def get_epoch_metrics_from_embedders(self, embedder_pair):
        ir_evaluator = self.get_ir_evaluator(embedder_pair)
        return ir_evaluator.evaluate()

    def get_metrics_df(self, embedder_pair, epoch):
        ir_evaluator = self.get_ir_evaluator(embedder_pair)
        metrics_dict = ir_evaluator.evaluate()
        return evaluation.get_single_metrics_df(
            f"{epoch}", metrics_dict["cos_sim"]
        ).iloc[0]

    def unconditionally_save_epoch_checkpoint(self, epoch, embedder_pair=None):
        if embedder_pair is None:
            embedder_pair = self.embedder_pair
        ir_evaluator = self.get_ir_evaluator(embedder_pair)

        if self.save_models:
            embedder_pair.query_embedder.save(
                os.path.join(self.save_dir, f"nbow_query_{epoch}")
            )
            embedder_pair.document_embedder.save(
                os.path.join(self.save_dir, f"nbow_document_{epoch}")
            )
            ir_evaluator.df.head().to_csv(
                os.path.join(self.save_dir, f"val_data{epoch}.csv")
            )
        metrics = ir_evaluator.evaluate()

        metrics_path = os.path.join(self.save_dir, f"metrics_{epoch}.yaml")
        print(yaml.dump(metrics["cos_sim"]))
        with open(metrics_path, "w") as f:
            yaml.dump(metrics, f)
