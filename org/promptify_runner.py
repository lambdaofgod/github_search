#!/usr/bin/env python3
from prompting import *
from mlutil.text import rwkv_utils
from promptify import OpenAI, Prompter
from promptify.models.nlp.model import Model as PromptifyModel
import fire
import numpy as np
import tqdm
import json

from record_writer import JsonWriterContextManager
from promptify_utils import PrompterWrapper

np.random.seed(seed=0)


def load_prompt_infos():
    pos_idxs = list(zip(range(0, 1000, 10), range(1000, 2000, 10)))
    pred_idxs = list(range(2000, 3000, 10))

    return [
        PromptInfo.from_df(train_nbow_df, list(pos), i)
        for (pos, i) in zip(pos_idxs, pred_idxs)
    ]


def run_progress_barred_loop(fn, record_writer_cls, inputs, writer_kwargs):
    with record_writer_cls(**writer_kwargs) as writer:
        for item in tqdm.tqdm(inputs):
            result = fn(item)
            writer.write_record(result)

    return writer


def main(
    model_path="/home/kuba/models/rwkv-4-raven-7b/RWKV-4-Raven-7B-v6-Eng-20230401-ctx4096.pth",
    prompt_template_name="md_prompt.jinja",
    templates_path="prompt_templates",
    out_path="/tmp/predictions.json",
    fake=False,
):
    writer_kwargs = {"file_path": out_path}
    prompter_wrapper = PrompterWrapper.create(
        model_path, templates_path, prompt_template_name, use_fake_model=fake
    )

    prompt_infos = load_prompt_infos()
    run_progress_barred_loop(
        prompter_wrapper.get_dict_with_generated_text,
        JsonWriterContextManager,
        prompt_infos,
        writer_kwargs,
    )


if __name__ == "__main__":
    fire.Fire(main)
