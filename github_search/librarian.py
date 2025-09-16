import jinja2
import pandas as pd
import random
import dspy
from typing import List
import ollama
from pydantic import BaseModel
import json
from typing import Any


def sample_context_repo_idxs(n_repos, n_shot) -> List[List[int]]:
    context_repo_idxs = []
    for i in range(n_repos):
        # Create a pool of indices for context, excluding the current index
        context_indices = list(range(n_repos))
        context_indices.pop(i)

        # Sample n_shot indices
        num_samples = min(n_shot, len(context_indices))
        sampled_indices = random.sample(context_indices, num_samples)
        context_repo_idxs.append(sampled_indices)
    return context_repo_idxs


def create_fewshot_prompts(
    repos_with_signatures_df, prompt_template, context_repo_idxs
):
    """
    Samples the records from `repos_with_signatures_df` and prepares few-shot prompts using Jinja template
    """
    template = jinja2.Template(prompt_template)
    prompts = []

    repo_records = repos_with_signatures_df.to_dict("records")

    for target_repo_record, sampled_indices in zip(repo_records, context_repo_idxs):
        context_repo_records = [repo_records[j] for j in sampled_indices]

        prompt = template.render(
            context_repo_records=context_repo_records,
            target_repo_record=target_repo_record,
        )
        prompts.append(prompt)

    return pd.Series(prompts, index=repos_with_signatures_df["repo"])


class LibrarianOutput(dspy.Signature):
    prompt: str = dspy.InputField()
    tasks: List[str] = dspy.OutputField()


class OllamaTypedPredict(BaseModel):
    model_name: str
    output_cls: Any

    def __call__(self, prompt):
        generated = ollama.chat(
            model=self.model_name,
            messages=[dict(role="user", content=prompt)],
            format=self.output_cls.model_json_schema(),
        )
        json_pred = json.loads(generated.message.content)
        return self.output_cls(**json_pred)

    class Config:
        arbitrary_types_allowed = True
