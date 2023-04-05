from prompting import PromptInfo
from promptify import Prompter
from mlutil.text import rwkv_utils
from pydantic import BaseModel, Field
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline
from pathlib import Path

from transformers import Text2TextGenerationPipeline
from typing import Any


# +
class HFModelWrapper(BaseModel):
    pipeline: Any
    
    def run(self, prompts, max_tokens):
        return self.pipeline(text_inputs=prompts,  max_new_tokens=max_tokens)
        
    class Config:
        arbitrary_types_allowed = True

class FakeModel:
    def run(self, *args, **kwargs):
        return ["foo"]


# -

def load_model(model_path):
    if "llama" in model_path:
        model_path = Path(model_path).expanduser()
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        bb_config = BitsAndBytesConfig(load_in_8bit=True)
        llama_model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=bb_config, device_map="auto"
        )
        return HFModelWrapper(pipeline=hf_pipeline(task="text-generation",model=llama_model, tokenizer=tokenizer))
    else:
        pipeline = rwkv_utils.RWKVPipelineWrapper.load(model_path=model_path)
        return rwkv_utils.RWKVPromptifyModel(pipeline=pipeline)


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
