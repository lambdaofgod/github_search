#!/usr/bin/env python3
# +
from prompting import *
from mlutil.text import rwkv_utils
from promptify import OpenAI, Prompter
from promptify.models.nlp.model import Model as PromptifyModel
from typing import Dict
import fire
import numpy as np
import tqdm
import json

from prompting_config import PromptDataConfig
# -

from record_writer import JsonWriterContextManager
from promptify_utils import PrompterWrapper
import clearml

np.random.seed(seed=0)



import ast

# +


class ContextLoader(BaseModel):
    
    data_path: str
    used_cols: List[str] = ["dependencies", "tasks", "repo"]
    truncation_kwargs: Dict[str, int] = {"dependencies": 10}
    
    def _truncate_fields(self, row):
        for field in self.truncation_kwargs.keys():
            row[field] = " ".join(row[field][:500].split(" ")[:self.truncation_kwargs[field]])

    def _get_pandas_dicts(self, df):
        for _, row in df.iterrows():
            row_dict = dict(row)
            self._truncate_fields(row_dict)
            yield row_dict
    
    def load_contexts(self, n_shots=2, n_samples=1000):
        df = pd.read_parquet(self.data_path).drop(["count"], axis=1)[self.used_cols]
        context_records = zip(*(
            self._get_pandas_dicts(df.sample(n_samples))
            for i in range(n_shots))
        )
        return context_records

    def load_predicted_df(self, n_samples=1000):
        df = pd.read_parquet(self.data_path).drop(["count"], axis=1)[self.used_cols]
        if n_samples is None: 
            return df
        else:
            return df.sample(n_samples)
    
    def load_prompt_infos(self, n_samples):
        predicted_df = self.load_predicted_df(n_samples=n_samples)
        predicted_records = self._get_pandas_dicts(predicted_df)
        contexts = self.load_contexts(n_samples=n_samples if n_samples is None else len(predicted_df))
        return (PromptInfo(repo_records=ctx, predicted_repo_record=dict(pred), true_tasks=ast.literal_eval(pred["tasks"])) for (ctx, pred) in zip(contexts, predicted_records))


# -

def run_progress_barred_loop(fn, record_writer_cls, inputs, total, writer_kwargs):
    with record_writer_cls(**writer_kwargs) as writer:
        for item in tqdm.tqdm(inputs, total=total):
            result = fn(item)
            writer.write_record(result)

    return writer


# +
from clearml import Dataset, Task
import jsonlines


class Main:
    
    def add_as_clearml_dataset(self, path, **kwargs):
        dataset = Dataset.create(**kwargs)
        dataset.sync_folder(local_path=path)
        dataset.upload()
        
    def add_clearml_task_output(self, task_name, output_file):
        task = Task(project_name="github_search_llms", task_name=task_name)
        task.upload_artifact('prediction_results', output_file)
        
    def sample_data(
        self,
        n_samples=100,
        pq_data_path="../output/nbow_data_test.parquet",
        out_data_path="../output/prompt_infos.jsonl",
        dset_kwargs=dict(dataset_project='github_search_llms', dataset_name='prompt_info_sample')
    ):
        prompt_infos = ContextLoader(data_path=pq_data_path).load_prompt_infos(n_samples=n_samples)
        pd.DataFrame.from_records(map(dict, prompt_infos)).to_json(out_data_path, orient="records", lines=True)
        self.add_as_clearml_dataset(out_data_path, dset_kwargs)

    def run_model(
        self,
        model_path="/home/kuba/Projects/forks/text-generation-webui/models/llama-13b-hf/",
        prompt_template_name="md_prompt.jinja",
        templates_path="prompt_templates",
        data_path="../output/prompt_infos.jsonl",
        out_dir="../output/llms/",
        n_samples=None,
        fake=False,
    ):
        print(f"loading data from {data_path}...")
        with jsonlines.open(data_path, 'r') as f:
            prompt_infos = [PromptInfo(**d) for d in f]
        model_nm = P(model_path).parent.name
        out_path = P(out_dir) / (model_nm + ".jsonl")
        out_path.parent.mkdir(exist_ok=True)
        if out_path.exists():
            out_path.unlink()
        print(f"write to {out_path}...")
        writer_kwargs = {"file_path": out_path}
        prompter_wrapper = PrompterWrapper.create(
            model_path, templates_path, prompt_template_name, use_fake_model=fake
        )

        run_progress_barred_loop(
            prompter_wrapper.get_dict_with_generated_text,
            JsonWriterContextManager,
            prompt_infos,
            total=n_samples,
            writer_kwargs=writer_kwargs,
        )
        self.add_clearml_task_output(f"prediction_{model_nm}", out_path)


# -

if __name__ == "__main__":
    fire.Fire(Main())

# !ls /home/kuba/Projects/forks/text-generation-webui/models/llama-13b-hf/
