import os

import pandas as pd
import torch
import pytorch_lightning as pl


from github_search.sentence_embeddings import pair_dataset, pair_models
import quaterion
from quaterion.dataset import PairsSimilarityDataLoader
from quaterion.eval.evaluator import Evaluator
from quaterion.eval.pair import RetrievalReciprocalRank, RetrievalPrecision
from quaterion.eval.samplers.pair_sampler import PairSampler

DATA_DIR = "data"


def run(model, train_dataset, val_dataset, params):
    use_gpu = params.get("cuda", torch.cuda.is_available())

    trainer = pl.Trainer(
        min_epochs=params.get("min_epochs", 1),
        max_epochs=params.get(
            "max_epochs", 10
        ),  # cache makes it possible to use a huge amount of epochs
        auto_select_gpus=use_gpu,
        log_every_n_steps=params.get(
            "log_every_n_steps", 10
        ),  # increase to speed up training
        gpus=int(use_gpu),
        num_sanity_val_steps=2,
    )
    train_dataloader = PairsSimilarityDataLoader(train_dataset, batch_size=1024)
    val_dataloader = PairsSimilarityDataLoader(val_dataset, batch_size=1024)
    quaterion.Quaterion.fit(model, trainer, train_dataloader, val_dataloader)

    metrics = {"rrk": RetrievalReciprocalRank(), "rp@1": RetrievalPrecision(k=1)}
    sampler = PairSampler()
    evaluator = Evaluator(metrics, sampler)
    results = quaterion.Quaterion.evaluate(
        evaluator, val_dataset, model.model
    )  # calculate metrics on the whole dataset to obtain more representative metrics values
    print(f"results: {results}")


# launch training
pl.seed_everything(42, workers=True)

df = pd.read_json("data/papers-with-abstracts.json.gz").dropna(
    subset=["title", "abstract"]
)

print(f"loaded {len(df)} papers with abstracts")
cols = ["title", "abstract"]
train_dataset = pair_dataset.PairDataset.from_pandas(df.iloc[:-10000], cols)
val_dataset = pair_dataset.PairDataset.from_pandas(df.iloc[-10000:], cols)
model = pair_models.PairSimilarityModel("models/rnn_abstract_readme_w2v")
run(model, train_dataset, val_dataset, {})
