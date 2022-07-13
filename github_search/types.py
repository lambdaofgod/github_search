import torch
from typing import TypeVar, List, Protocol, Generic

T = TypeVar("T")


class Encoder(Protocol, Generic[T]):
    """
    encoder that generalizes SentenceTransformers - Encoder[str]
    to encoders that can handle data with other types like graphs
    """

    def encode(
        self,
        inputs: List[T],
        show_progress_bar: bool = False,
        batch_size: int = 32,
        convert_to_tensor: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """encode raw data"""
