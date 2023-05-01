from dataclasses import dataclass, asdict, field
from mlutil import chatgpt_api
import numpy as np
import pandas as pd
import sys
from pathlib import Path

from prompting import PromptInfo, Prompts
import json
import fire
import logging

logging.basicConfig(level="INFO")


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


from typing import List
from pathlib import Path as P

base_prompt = """
repository {}
contains files {}
its tags are {}
"""


repo_records = get_repo_records_by_index(train_nbow_df, [5, 10])
other_repo_record = get_repo_records_by_index(train_nbow_df, [1])[0]


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


from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
import torch

from transformers import BitsAndBytesConfig


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


@dataclass
class TextGenerator:

    tokenizer: AutoTokenizer
    model: AutoModelForCausalLM
    prompt_template: str = field(default=Prompts.heading_prompt)

    def encode(self, prompts, max_length=None, add_special_tokens=True):
        return self.tokenizer.encode(
            prompts,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            add_special_tokens=True,
        )

    def generate(self, in_texts):
        ids = self.encode(in_texts).to(device="cuda")
        generated_ids = self.model.generate(
            ids, **generation_args
        )  # , **generation_args)
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    def get_batches(self, lst: List, batch_size: int) -> List:
        results = []
        for i in tqdm.tqdm(range(0, len(lst), batch_size)):
            batch = lst[i : i + batch_size]
            results.append(batch)
        return results

    def process_batch(self, prompt_infos):
        generated_texts = [
            self.generate(p.format_prompt(self.prompt_template))
            for p in prompt_infos[:1]
        ]
        return [
            {
                "prompt_repos": [rec["repo"] for rec in p.repo_records],
                "repo": p.predicted_repo_record["repo"],
                "generated_text": gen_text,
            }
            for (p, gen_text) in zip(prompt_infos, generated_texts)
        ]


text_generation_path = "/home/kuba/Projects/forks/text-generation-webui"
model_name = "llama-13b-hf"
model_path = (
    "google/flan-t5-large"  # Path(f"{text_generation_path}/models/{model_name}")
)


def main(model_name_or_path):

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    bb_config = BitsAndBytesConfig(load_in_8bit=True)

    logging.info(f"using {model_name_or_path}")
    if "llama" in P(model_name_or_path).name:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, quantization_config=bb_config, device_map="auto"
        )

    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path, device_map="auto"
        )

    generator = TextGenerator(tokenizer, model)

    with torch.no_grad():

        file_prefix = P(model_name_or_path).name
        with open(f"{file_prefix}_generated.jsonl", "w") as f:
            for i, batch_prompt_infos in enumerate(
                tqdm.tqdm(generator.get_batches(prompt_infos, 1))
            ):

                generated_record = generator.process_batch(batch_prompt_infos)
                if i == 0:
                    print("#" * 50)
                    print("## GENERATED")
                    print(generated_record[0]["generated_text"])
                f.write(json.dumps(generated_record[0]))
                f.write("\n")


if __name__ == "__main__":
    fire.Fire(main)
# ids = tokenizer.batch_encode_plus(texts)
# generated_ids = llama.generate(ids)
# tokenizer.batch_decode(generated_ids)
