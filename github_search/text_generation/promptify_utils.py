from prompting import PromptInfo
from promptify import Prompter
from mlutil.text import rwkv_utils
from pydantic import BaseModel, Field
from transformers import (
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline as hf_pipeline,
)
from pathlib import Path
import torch

from transformers import Text2TextGenerationPipeline
from typing import Any


class HFModelWrapper:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @staticmethod
    def load_from_path(model_path):
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16
        )
        model.cuda()
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return HFModelWrapper(model, tokenizer)

    def run(self, prompts, max_tokens):
        inputs = self.tokenizer.encode(prompts[0], return_tensors="pt").cuda()
        input_length = inputs.shape[-1]
        outputs = self.model.generate(inputs, max_new_tokens=max_tokens)[
            :, input_length:
        ]
        return [self.tokenizer.decode(outputs[0])]

    @staticmethod
    def create_llama(model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        bb_config = BitsAndBytesConfig(load_in_8bit=True)
        llama_model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=bb_config, device_map="auto"
        )
        return HFModelWrapper(llama_model, tokenizer)


class FakeModel:
    def run(self, *args, **kwargs):
        return ["foo"]


def load_model(model_path):
    if "llama" in model_path:
        return HFModelWrapper.create_llama(model_path)
    elif "rwkv" in model_path:
        pipeline = rwkv_utils.RWKVPipelineWrapper.load(model_path=model_path)
        return rwkv_utils.RWKVPromptifyModel(pipeline=pipeline)
    else:
        return HFModelWrapper.load_from_path(model_path)


class PrompterWrapper(BaseModel):
    prompter: Prompter
    template_name: str
    max_tokens: int = Field(default=20)

    def generate_text_from_promptinfo(self, pinfo: PromptInfo):
        promptify_args = pinfo.get_promptify_input_dict()
        return self.prompter.fit(
            template_name=self.template_name,
            max_tokens=self.max_tokens,
            **promptify_args
        )

    def get_dict_with_generated_text(self, pinfo: PromptInfo):
        out_record = dict(pinfo)
        generated_text = self.generate_text_from_promptinfo(pinfo)
        out_record["generated_text"] = generated_text
        return out_record

    class Config:
        arbitrary_types_allowed = True

    @staticmethod
    def create(
        model_path, templates_path, template_name, max_tokens=20, use_fake_model=False
    ):
        if use_fake_model:
            model = FakeModel()
        else:
            model = load_model(model_path)
        nlp_prompter = Prompter(model, templates_path=templates_path)
        return PrompterWrapper(
            prompter=nlp_prompter, template_name=template_name, max_tokens=max_tokens
        )
