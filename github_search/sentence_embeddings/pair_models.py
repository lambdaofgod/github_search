import os

from quaterion import TrainableModel
from quaterion.eval.attached_metric import AttachedMetric
from quaterion.eval.pair import RetrievalPrecision, RetrievalReciprocalRank
from quaterion.loss import MultipleNegativesRankingLoss
from quaterion.train.cache import CacheConfig, CacheType
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Pooling, Transformer
from torch import Tensor, nn
from torch.optim import Adam

from quaterion_models.encoders import Encoder
from quaterion_models.heads.skip_connection_head import SkipConnectionHead
from quaterion_models.types import CollateFnType, TensorInterchange


class PairEncoder(Encoder):
    def __init__(self, transformer):
        super().__init__()
        self.encoder = transformer

    @property
    def trainable(self) -> bool:
        # Defines if we want to train encoder itself, or head layer only
        return False

    @property
    def embedding_size(self) -> int:
        return self.encoder.get_sentence_embedding_dimension()

    def forward(self, batch: TensorInterchange) -> Tensor:
        return self.encoder(batch)["sentence_embedding"]

    def get_collate_fn(self) -> CollateFnType:
        # `collate_fn` is a function that converts input samples into Tensor(s) for use as encoder input.
        return self.encoder.tokenize

    def save(self, output_path: str):
        os.makedirs(output_path, exist_ok=True)
        self.encoder.save(output_path)

    @classmethod
    def load(cls, input_path: str) -> Encoder:
        transformer = SentenceTransformer(input_path)
        return PairEncoder(transformer)

class PairSimilarityModel(TrainableModel):
    def __init__(self, model_path_or_name, lr=1e-3, *args, **kwargs):
        self.lr = lr
        self.model_path_or_name = model_path_or_name
        super().__init__(*args, **kwargs)

    def configure_metrics(self):
        # attach batch-wise metrics which will be automatically computed and logged during training
        return [
            AttachedMetric(
                "RetrievalPrecision",
                RetrievalPrecision(k=1),
                prog_bar=True,
                on_epoch=True,
            ),
            AttachedMetric(
                "RetrievalReciprocalRank",
                RetrievalReciprocalRank(),
                prog_bar=True,
                on_epoch=True,
            ),
        ]

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=self.lr)

    def configure_loss(self):
        # `symmetric` means that we take into account correctness of both the closest answer to a question and the closest question to an answer
        return MultipleNegativesRankingLoss(symmetric=True)

    def configure_encoders(self):
        pre_trained_model = SentenceTransformer(self.model_path_or_name)
        encoder = PairEncoder(pre_trained_model)
        return encoder

    def configure_head(self, input_embedding_size: int):
        return SkipConnectionHead(input_embedding_size)

    def configure_caches(self):
        # Cache stores frozen encoder embeddings to prevent repeated calculations and increase training speed.
        # AUTO preserves the current encoder's device as storage, batch size does not affect training and is used only to fill the cache before training.
        return CacheConfig(CacheType.AUTO, batch_size=256)
