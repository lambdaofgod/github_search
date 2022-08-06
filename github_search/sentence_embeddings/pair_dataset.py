import json
from typing import List, Dict
from torch.utils.data import Dataset
from quaterion.dataset.similarity_samples import SimilarityPairSample

class PairDataset(Dataset):

    def __init__(self, dataset : List[Dict[str, str]]):
        self.dataset = dataset

    def __getitem__(self, index) -> SimilarityPairSample:
        line = self.dataset[index]
        question = line["text"]
        # All questions have a unique subgroup
        # Meaning that all other answers are considered negative pairs
        subgroup = hash(question)
        score = 1
        return SimilarityPairSample(
            obj_a=question, obj_b=line["other_text"], score=score, subgroup=subgroup
        )

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def read_dataset(dataset_path) -> List[Dict[str, str]]:
        """Read jsonl-file into a memory."""
        with open(dataset_path, "r") as fd:
            return [json.loads(json_line) for json_line in fd]

    @staticmethod
    def from_pandas(df, columns):
        df = df[columns]
        df.columns = ["text", "other_text"]
        return PairDataset([
            record.to_dict()
            for __, record in df.iterrows()
        ])

    @staticmethod
    def from_path(path):
        return PairDataset(PairDataset.read_dataset(path))
