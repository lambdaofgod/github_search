#!/usr/bin/env python3
from prompting import *
from mlutil.text import rwkv_utils
from promptify import OpenAI, Prompter
from promptify.models.nlp.model import Model as PromptifyModel


def load_model(model_path):
    return rwkv_utils.RWKVPipelineWrapper.load(model_path=model_path)


class fakemodel:
    def run(self, *args, **kwargs):
        return ["foo"]


def setup_prompter(
    model_path="/home/kuba/models/rwkv-4-raven-7b/RWKV-4-Raven-7B-v6-Eng-20230401-ctx4096.pth",
    templates_path="prompt_templates",
):
    nlp_prompter = Prompter(fakemodel(), templates_path=templates_path)
    return nlp_prompter


if __name__ == "__main__":
    prompt_template_name = "md_prompt.jinja"
    pos_idxs = list(zip(range(0, 1000, 10), range(1000, 2000, 10)))
    pred_idxs = list(range(2000, 3000, 10))

    prompt_infos = [
        PromptInfo.from_df(train_nbow_df, list(pos), i)
        for (pos, i) in zip(pos_idxs, pred_idxs)
    ]
    promptify_args = prompt_infos[0].get_promptify_input_dict()
    print(promptify_args)
    nlp_prompter = setup_prompter()
    print(nlp_prompter.fit(template_name=prompt_template_name, **promptify_args))
