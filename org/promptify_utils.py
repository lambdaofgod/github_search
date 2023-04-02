from prompting import PromptInfo
from promptify import Prompter
from mlutil.text import rwkv_utils
from pydantic import BaseModel, Field


def load_model(model_path):
    pipeline = rwkv_utils.RWKVPipelineWrapper.load(model_path=model_path)
    return rwkv_utils.RWKVPromptifyModel(pipeline=pipeline)


class FakeModel:
    def run(self, *args, **kwargs):
        return ["foo"]


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
