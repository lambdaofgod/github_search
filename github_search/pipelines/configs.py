from pydantic import BaseModel, Field
from typing import List


class EvaluationConfig(BaseModel):
    id_col: str = Field(default="repo")
    reference_text_col: str = Field(default="true_text")
    metric_names: List[str] = Field(
        default_factory=lambda: [
            "edit_word",
            "jaccard_lst",
            "bleurt",
            "rouge",
            # "wmd",
            "sentence_transformer_similarity",
        ]
    )
