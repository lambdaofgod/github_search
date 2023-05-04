from promptify import Prompter
from mlutil.text import rwkv_utils
from pydantic import BaseModel, Field
from transformers import (
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline as hf_pipeline,
)
from mlutil import minichain_utils
from pathlib import Path
import torch

import minichain
from minichain.backend import Backend

from transformers import Text2TextGenerationPipeline
from typing import Any, Union


from github_search.text_generation.prompting import PromptInfo


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


import abc


class PrompterWrapper(abc.ABC):
    prompter: Prompter
    template_name: str
    max_tokens: int = Field(default=20)

    @abc.abstractmethod
    def generate_text_from_promptinfo(self, pinfo: PromptInfo):
        pass

    def get_dict_with_generated_text(self, pinfo: PromptInfo):
        out_record = dict(pinfo)
        generated_text = self.generate_text_from_promptinfo(pinfo)
        out_record["generated_text"] = generated_text
        return out_record


class MinichainModelConfig(abc.ABC):
    pass


class MinichainHFConfig(MinichainModelConfig):
    model_name_or_path: str
    device: int = Field(default=dict(device=0))


class MinichainRWKVConfig(MinichainModelConfig):
    model_name_or_path: str


class MinichainPrompterWrapper(PrompterWrapper, BaseModel):
    prompt_fn: Callable[[PromptInfo], str]

    def generate_text_from_promptinfo(self, pinfo: PromptInfo):
        promptify_args = pinfo.get_promptify_input_dict()
        return self.prompter.fit(
            template_name=self.template_name,
            max_tokens=self.max_tokens,
            **promptify_args
        )

    @classmethod
    def create(
        cls,
        model: Backend,
        prompt_template: Optional[str],
        prompt_template_path: Optional[str],
    ):
        @prompt(
            model,
            **cls.make_minichain_template_kwargs(prompt_template, prompt_template_path)
        )
        def prompt_fn(model, prompt_info: PromptInfo):
            return model(dict(prompt_info))

        return MinichainPrompterWrapper(prompt_fn=prompt_fn)

    @classmethod
    def make_minichain_prompt_kwargs(
        cls, prompt_template: Optional[str], prompt_template_path: Optional[str]
    ):
        if prompt_template_path is not None:
            return dict(prompt_template_path=prompt_template_path)
        else:
            assert prompt_template is None
            return dict(prompt_template=prompt_template)

    @classmethod
    def load_model(cls, config: MinichainModelConfig):
        if config.is_rwkv:
            return minichain_utils.RWKVModel.load(config.model_name_or_path)
        else:
            return minichain_utils.HuggingFaceLocalModel.load(config.model_name_or_path)


class PromptifyPrompterWrapper(PrompterWrapper):
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
