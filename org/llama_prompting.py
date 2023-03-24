from dataclasses import dataclass, asdict
from mlutil import chatgpt_api
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# sys.path.insert(0, str(Path("/home/kuba/Projects/forks/GPTQ-for-LLaMa")))
# import llama


api_key_path = (
    "~/Projects/org/openai_key.txt"  # specify file path if OPENAI_API_KEY is not in env
)
chatgpt_client = chatgpt_api.ChatGPTClient(api_key_path)
"initialized api"

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline


def get_n_tokens(text):
    ids = tokenizer(text)["input_ids"]
    return len(ids)


import pandas as pd

train_nbow_df = pd.read_parquet("../output/nbow_data_train.parquet").drop(
    ["count"], axis=1
)
train_nbow_df.head()


def preprocess_dep(dep):
    return P(dep).name


def select_deps(deps, n_deps):
    return [preprocess_dep(dep) for dep in deps if not "__init__" in dep][:n_deps]


def get_repo_records_by_index(
    data_df, indices, fields=["repo", "dependencies", "tasks"], n_deps=10
):
    records_df = data_df.iloc[indices].copy()
    raw_deps = records_df["dependencies"].str.split()
    records_df["dependencies"] = raw_deps.apply(lambda dep: select_deps(dep, n_deps))
    return records_df[fields].to_dict(orient="records")


from typing import List
from pathlib import Path as P

base_prompt = """
repository {}
contains files {}
its tags are {}
"""


@dataclass
class PromptInfo:
    """
    information about sample repositories passed to prompt
    """

    repo_records: List[dict]
    predicted_repo_record: dict

    def get_single_prompt(self, record):
        repo = record["repo"]
        dependencies = ", ".join(record["dependencies"])
        tasks = record["tasks"]
        return base_prompt.format(repo, dependencies, tasks)

    def get_prompt(self):
        prefix_prompt = "\n".join(
            self.get_single_prompt(record) for record in self.repo_records
        )
        (
            other_repo_name,
            other_repo_filenames,
            other_repo_tasks,
        ) = self.predicted_repo_record.values()
        other_repo_filenames = [P(fname).name for fname in other_repo_filenames]
        return (
            prefix_prompt
            + f"\nrepository {other_repo_name}\n"
            + f"contains files: {', '.join(other_repo_filenames)}\n"
            + "tags: "
        )

    @classmethod
    def from_df(cls, data_df, pos_indices, pred_index, n_deps=10):
        return PromptInfo(
            get_repo_records_by_index(data_df, pos_indices, n_deps=n_deps),
            get_repo_records_by_index(data_df, [pred_index], n_deps=n_deps)[0],
        )


repo_records = get_repo_records_by_index(train_nbow_df, [5, 10])
other_repo_record = get_repo_records_by_index(train_nbow_df, [1])[0]

prompt_info = PromptInfo(repo_records, other_repo_record)
prompt = prompt_info.get_prompt()
prompt


def get_sample_prompt_info(data_df, n_labeled):
    sample_labeled_indices = np.random.randint(2)
    repo_records = get_repo_records_by_index(train_nbow_df, [100, 101])
    other_repo_record = get_repo_records_by_index(train_nbow_df, [20])[0]

    prompt_info = PromptInfo(repo_records, other_repo_record)
    return prompt_info


pos_idxs = list(zip(range(0, 1000, 10), range(1000, 2000, 10)))
pred_idxs = list(range(2000, 3000, 10))

prompt_infos = [
    PromptInfo.from_df(train_nbow_df, list(pos), i)
    for (pos, i) in zip(pos_idxs, pred_idxs)
]

true_tasks = [pinfo.predicted_repo_record["tasks"] for pinfo in prompt_infos]
true_tasks[0]

import sys
from pathlib import Path

# sys.path.insert(0, "modules")


from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch

text_generation_path = "/home/kuba/Projects/forks/text-generation-webui"
model_name = "llama-13b-hf"
model_path = Path(f"{text_generation_path}/models/{model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_path)

from transformers import BitsAndBytesConfig

bb_config = BitsAndBytesConfig(load_in_8bit=True)


from typing import List, Callable
import tqdm


generation_args = {
    "use_cache": True,
    "max_new_tokens": 20,
    "eos_token_id": [2],
    "stopping_criteria": [],
    "do_sample": True,
    "temperature": 1.0,
    "top_p": 0.18,
    "typical_p": 1,
    "repetition_penalty": 1.15,
    "encoder_repetition_penalty": 1,
    "top_k": 30,
    "min_length": 0,
    "no_repeat_ngram_size": 0,
    "num_beams": 1,
    "penalty_alpha": 0,
    "length_penalty": 1,
    "early_stopping": False,
}


def get_max_prompt_length(n_tokens):
    max_length = 2048 - n_tokens
    # if shared.soft_prompt:
    #    max_length -= shared.soft_prompt_tensor.shape[1]
    return max_length


def encode(prompts, max_length=None, add_special_tokens=True):
    return tokenizer.encode(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
        add_special_tokens=True,
    )


def generate(in_texts):
    ids = encode(in_texts).to(device="cuda")
    generated_ids = llama_model.generate(ids, **generation_args)  # , **generation_args)
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


llama_model = AutoModelForCausalLM.from_pretrained(
    model_path, quantization_config=bb_config, device_map="auto"
)


def get_batches(lst: List, batch_size: int) -> List:
    results = []
    for i in tqdm.tqdm(range(0, len(lst), batch_size)):
        batch = lst[i : i + batch_size]
        results.append(batch)
    return results


def process_batch(prompt_infos):
    generated_texts = [generate(p.get_prompt()) for p in prompt_infos[:1]]
    return [
        {
            "prompt_repos": [rec["repo"] for rec in p.repo_records],
            "repo": p.predicted_repo_record["repo"],
            "generated_text": gen_text,
        }
        for (p, gen_text) in zip(prompt_infos, generated_texts)
    ]


import json

with torch.no_grad():
    with open("llama_generated.txt", "w") as f:
        for batch_prompt_infos in tqdm.tqdm(get_batches(prompt_infos, 1)):

            generated_record = process_batch(batch_prompt_infos)
            f.write(json.dumps(generated_record[0]))
            f.write("\n")

# ids = tokenizer.batch_encode_plus(texts)
# generated_ids = llama.generate(ids)
# tokenizer.batch_decode(generated_ids)
