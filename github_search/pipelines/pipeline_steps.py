;; [[file:pipeline_steps.org::+BEGIN_SRC elisp :session pipeline_steps.org :exports both :comments link][No heading:1]]
(require 'mmm-auto)

(setq mmm-global-mode 'maybe)
(mmm-add-mode-ext-class 'org-mode "\\.py\\'" 'org-py)
;; No heading:1 ends here

# [[file:pipeline_steps.org::*Running pipeline step by step][Running pipeline step by step:1]]
import pandas as pd
from github_search.pipelines.steps import sample_data_step, expand_documents_step, evaluate_generated_texts_step, evaluate_generated_texts
from tgutil.configs import PipelineConfig, ConfigPaths, APIConfig, TextGenerationConfig, SamplingConfig, PromptConfig
from tgutil.prompting_runner import sample_data, expand_documents
import logging
import yaml
# Running pipeline step by step:1 ends here

# [[file:pipeline_steps.org::*Running pipeline step by step][Running pipeline step by step:2]]
def load_config_yaml_key(cls, config_path, key):
    """
    loads appropriate config from path
    the yaml file should contain 'key' and the 'cls' object will be created from its value
    """
    with open(config_path) as f:
        conf = yaml.safe_load(f)[key]
    return cls(**conf)
# Running pipeline step by step:2 ends here

# [[file:pipeline_steps.org::*Running pipeline step by step][Running pipeline step by step:3]]
logging.basicConfig(level="INFO")

sampling = "no_sampling"
generation_method = "api_rwkv"

cfg_path = "conf/text_generation/config.yaml"
generation_config = load_config_yaml_key(APIConfig, "conf/generation.yaml", generation_method)
sampling_config = load_config_yaml_key(SamplingConfig, "conf/sampling.yaml", sampling)
prompt_config = load_config_yaml_key(PromptConfig, "conf/prompts.yaml", "few_shot_markdown")
#cfg = PipelineConfig.load(cfg_path)
# Running pipeline step by step:3 ends here

# [[file:pipeline_steps.org::*Running pipeline step by step][Running pipeline step by step:4]]
model_type = generation_config.model_name
# Running pipeline step by step:4 ends here

# [[file:pipeline_steps.org::*Running pipeline step by step][Running pipeline step by step:5]]
cfg
# Running pipeline step by step:5 ends here

# [[file:pipeline_steps.org::*Sample][Sample:1]]
prompt_infos = sample_data(sampling_config)
# Sample:1 ends here

# [[file:pipeline_steps.org::*Sample][Sample:2]]
prompt_infos[:1]
# Sample:2 ends here

# [[file:pipeline_steps.org::*Expand documents (generate texts)][Expand documents (generate texts):1]]
generated_records = expand_documents(generation_config, prompt_config, prompt_infos)
# Expand documents (generate texts):1 ends here

# [[file:pipeline_steps.org::*Expand documents (generate texts)][Expand documents (generate texts):2]]
generated_records
# Expand documents (generate texts):2 ends here

# [[file:pipeline_steps.org::*Expand documents (generate texts)][Expand documents (generate texts):3]]
generated_records.to_json(f"output/{model_type}_generated_records.json", orient="records", lines=True)
# Expand documents (generate texts):3 ends here

# [[file:pipeline_steps.org::*Evaluate text generation][Evaluate text generation:1]]
generated_records = pd.read_json(f"output/{generation_method}_generated_records_{sampling}.json", orient="records", lines=True)
# Evaluate text generation:1 ends here

# [[file:pipeline_steps.org::*Evaluate text generation][Evaluate text generation:2]]
generated_records["repo"] = generated_records["predicted_repo_record"].apply(lambda rec: rec["repo"])
generated_records["tasks"] = generated_records["true_tasks"]
generated_records.columns
# Evaluate text generation:2 ends here

# [[file:pipeline_steps.org::*Evaluate text generation][Evaluate text generation:3]]
evaluated_df = evaluate_generated_texts(generated_records[["repo", "generated_text", "tasks"]], "../../data/paperswithcode_with_tasks.csv")
# Evaluate text generation:3 ends here

# [[file:pipeline_steps.org::*Evaluate text generation][Evaluate text generation:4]]
evaluated_df.to_json(f"output/{model_type}_evaluated_records.json", orient="records", lines=True)
# Evaluate text generation:4 ends here

# [[file:pipeline_steps.org::*Results][Results:1]]
evaluated_df.describe()
# Results:1 ends here

# [[file:pipeline_steps.org::*Evaluate information retrieval][Evaluate information retrieval:1]]
evaluated_df = pd.read_json(f"output/{model_type}_evaluated_records.json", orient="records", lines=True)
# Evaluate information retrieval:1 ends here

# [[file:pipeline_steps.org::*Evaluate information retrieval][Evaluate information retrieval:2]]
evaluated_df["reference_text"]
# Evaluate information retrieval:2 ends here

# [[file:pipeline_steps.org::*Evaluate information retrieval][Evaluate information retrieval:3]]
def replace_list_chars(text):
    return text.replace("[", "").replace("]", "").replace(",", "").replace("'", "")

def process_generated_text(text):
    return replace_list_chars(text.strip().split("\n")[0])
# Evaluate information retrieval:3 ends here

# [[file:pipeline_steps.org::*Evaluate information retrieval][Evaluate information retrieval:4]]
max_len = 100
ir_df = pd.read_parquet("../../output/nbow_data_test.parquet")[["repo", "dependencies"]].merge(evaluated_df, on="repo")
ir_df = ir_df.assign(generated_tasks=ir_df["generated_text"].apply(process_generated_text))
ir_df = ir_df.assign(truncated_dependencies=ir_df["dependencies"].apply(lambda doc: " ".join(doc.split()[:max_len])))
# Evaluate information retrieval:4 ends here

# [[file:pipeline_steps.org::*Evaluate information retrieval][Evaluate information retrieval:5]]
processed_text = ir_df["generated_text"].apply(process_generated_text).iloc[0]
processed_text
# Evaluate information retrieval:5 ends here

# [[file:pipeline_steps.org::*Evaluate information retrieval][Evaluate information retrieval:6]]
from github_search.ir.evaluator import InformationRetrievalEvaluatorConfig, EmbedderPairConfig, InformationRetrievalColumnConfig
from github_search.ir import evaluator, models
import yaml


with open("conf/ir_config_nbow.yaml") as f:
    ir_config = InformationRetrievalEvaluatorConfig(**yaml.safe_load(f))
# Evaluate information retrieval:6 ends here

# [[file:pipeline_steps.org::*Evaluate information retrieval][Evaluate information retrieval:7]]
ir_evaluator = evaluator.InformationRetrievalEvaluator.setup_from_df(ir_df, ir_config)
ir_results = ir_evaluator.evaluate()
# Evaluate information retrieval:7 ends here

# [[file:pipeline_steps.org::*Evaluate information retrieval][Evaluate information retrieval:8]]
import pprint

pprint.pprint(ir_results)
# Evaluate information retrieval:8 ends here

# [[file:pipeline_steps.org::*Evaluate information retrieval][Evaluate information retrieval:9]]
import pprint

pprint.pprint(ir_results)
# Evaluate information retrieval:9 ends here

# [[file:pipeline_steps.org::*Evaluate information retrieval][Evaluate information retrieval:10]]
import pprint

pprint.pprint(ir_results)
# Evaluate information retrieval:10 ends here

# [[file:pipeline_steps.org::*Comparing IR to text generation metrics][Comparing IR to text generation metrics:1]]
(ir_df["generated_text"] + ir_df["dependencies"]).iloc[0]
# Comparing IR to text generation metrics:1 ends here
