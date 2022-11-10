from dataclasses import dataclass, field

import torch
import numpy as np
import tqdm
from github_search.neural_bag_of_words.models import NBOWLayer
from github_search.neural_bag_of_words.data import (
    NBOWNumericalizer,
    pack_sequences_as_tensors,
)


@dataclass
class NBOWEmbedder:

    nbow_layer: NBOWLayer
    numericalizer: NBOWNumericalizer
    max_len: int = field(default=200)

    def encode(
        self, texts, batch_size=32, show_progress_bar=False, convert_to_tensor=True
    ):
        nums = self.numericalizer.numericalize_texts(texts)
        idx_iter = tqdm.tqdm(range(0, len(nums), batch_size))
        if show_progress_bar:
            idx_iter = tqdm.tqdm(idx_iter)

        num_tensors_iter = (
            pack_sequences_as_tensors(
                nums[i : i + batch_size], self.numericalizer.get_padding_idx()
            )[:, : self.max_len]
            for i in idx_iter
        )
        with torch.no_grad():
            embeddings = [
                self.nbow_layer(b.to(self.nbow_layer.embedding.weight.device))
                .cpu()
                for b in num_tensors_iter
            ]
        if convert_to_tensor:
            return torch.row_stack(embeddings)
        return np.row_stack([t.numpy() for t in embeddings])
