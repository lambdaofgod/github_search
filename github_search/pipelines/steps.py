from typing import List, Tuple
import os
from pathlib import Path

import pandas as pd
from github_search.pipelines.postprocessing import GenerationPostprocessor
from github_search.python_code import code_selection
from typing_extensions import Annotated
import logging
import ast
from jinja2 import Environment, FileSystemLoader
import tqdm

try:
    from tgutil.configs import PromptConfig, load_config_from_dict
    from tgutil.prompting import ContextPromptInfo
    from tgutil.prompting_runner import DocumentExpander
except:
    logging.warning("tgutil not installed, using local imports")
import dspy
from github_search.samplers import TaskSizeRepoSampler
from github_search.lms.code2documentation import (
    run_code2doc_on_df,
    run_code2doc_on_files_df,
)
from phoenix.otel import register
from opentelemetry import trace


class ZenMLSteps:
    @staticmethod
    def expand_documents_step(
        text_generation_config: dict, prompt_config: dict, prompt_infos: List[dict]
    ) -> Tuple[
        Annotated[pd.DataFrame, "raw_generated_texts_df"],
        Annotated[List[dict], "failures"],
    ]:
        logging.info("expanding documents")
        logging.info(f"using text generation config: {text_generation_config}")
        logging.info(f"using prompt config: {prompt_config}")

        # Load prompt template
        templates_path = prompt_config["templates_path"]
        template_name = prompt_config["prompt_template_name"]

        env = Environment(loader=FileSystemLoader(templates_path))
        template = env.get_template(template_name)
        # configure the Phoenix tracer
        tracer_provider = register(project_name="librarian")
        # Initialize ollama
        lm = dspy.OllamaLocal(
            model=text_generation_config.get("model_name", "llama2"),
            base_url=text_generation_config.get("base_url", "http://localhost:11434"),
        )
        dspy.configure(lm=lm)

        generated_texts = []
        failures = []

        tracer = trace.get_tracer("librarian")

        # Iterate over records and generate text
        for i, record in tqdm.tqdm(enumerate(prompt_infos), total=len(prompt_infos)):
            try:
                # Fill the prompt template with the record data
                filled_prompt = template.render(**record)
                if i == 0:
                    logging.info(f"Filled prompt for record {i}")
                    logging.info(filled_prompt)

                # Generate text using ollama
                for gen_idx in range(text_generation_config.get("n_generations", 1)):
                    try:
                        response = lm(filled_prompt)
                        if type(response) is list:
                            response = response[0]
                        generated_texts.append(
                            {
                                "prompt_info": record,
                                "generated_text": response,
                                "prompt": filled_prompt,
                                "generation_index": gen_idx,
                                "record_index": i,
                            }
                        )
                        with tracer.start_as_current_span(
                            f"llm_call_to_model_{i}_{gen_idx}"
                        ) as llm_span:
                            llm_span.set_attribute("prompt", filled_prompt)
                            llm_span.set_attribute("response", response)

                    except Exception as e:
                        logging.error(
                            f"Failed to generate text for record {i}, generation {gen_idx}: {e}"
                        )
                        failures.append(
                            {
                                "record_index": i,
                                "generation_index": gen_idx,
                                "error": str(e),
                                "record": record,
                            }
                        )
            except Exception as e:
                logging.error(f"Failed to process record {i}: {e}")
                failures.append({"record_index": i, "error": str(e), "record": record})

        raw_generated_texts_df = pd.DataFrame(generated_texts)
        return raw_generated_texts_df, failures

    @staticmethod
    def postprocess_generated_texts(
        raw_generated_texts_df: pd.DataFrame,
    ) -> Annotated[pd.DataFrame, "generated_texts_df"]:
        return GenerationPostprocessor.run(raw_generated_texts_df)

    @staticmethod
    def sample_data_step(
        prompt_config: dict, sampling_config: dict
    ) -> Annotated[List[dict], "prompt_infos"]:
        from tgutil.configs import PromptConfig, SamplingConfig
        from tgutil.prompting_runner import DocumentExpander

        sampling_config = SamplingConfig(**sampling_config)
        prompt_config = PromptConfig(**prompt_config)
        return [
            pinfo.dict()
            for pinfo in DocumentExpander.sample_data(prompt_config, sampling_config)
        ]


class Code2DocSteps:
    @staticmethod
    def prepare_data_df(repos_df_path, python_code_path):
        logging.info("loading repo data from %s", repos_df_path)
        repos_df = pd.read_json(repos_df_path)
        repos_df = repos_df[~repos_df["readme"].isna()]
        logging.info("loading python code data from %s", python_code_path)
        python_code_df = pd.read_parquet(python_code_path)
        repos_with_all_data_df = repos_df[
            repos_df["repo"].isin(python_code_df["repo_name"])
        ]
        logging.info("sampling Python files per repo")
        selected_python_code_df = code_selection.get_python_files_with_selected_code_df(
            python_code_df
        )
        return repos_with_all_data_df, selected_python_code_df

    @staticmethod
    def prepare_data(
        repos_df_path, python_code_path, repos_output_path, selected_python_code_path
    ):
        repos_with_all_data_df, selected_python_code_df = Code2DocSteps.prepare_data_df(
            repos_df_path, python_code_path
        )
        repos_with_all_data_df.to_json(repos_output_path, orient="records", lines=True)
        # for some reason there are errors in parquet so we'll save it to feather
        selected_python_code_df.to_feather(selected_python_code_path)

    @staticmethod
    def create_repos_sample_df(
        repos_df,
        python_code_df,
        n_repos_per_task=10,
        min_task_size=5,
        max_task_size=500,
        max_random_baseline_score=0.5,
        max_tasks_per_repo=10,
    ):
        python_code_df = python_code_df.dropna(subset=["selected_code"])
        repos_df = repos_df[repos_df["repo"].isin(python_code_df["repo_name"])]
        repos_df["tasks"] = repos_df["tasks"].apply(ast.literal_eval)
        repos_df = repos_df[repos_df["tasks"].apply(len) <= max_tasks_per_repo]
        repo_sampler = TaskSizeRepoSampler(
            min_task_count=min_task_size,
            n_repos_per_task=n_repos_per_task,
            max_repos_per_task=max_task_size,
        )
        return repo_sampler.get_sampled_task_repos_df(
            repos_df, max_baseline_score=max_random_baseline_score
        )

    @staticmethod
    def create_repos_sample(
        repos_df_path,
        selected_python_code_path,
        output_path,
        n_repos_per_task=10,
        min_task_size=5,
        max_task_size=500,
        max_random_baseline_score=0.5,
    ):
        logging.info("Creating repos sample")
        repos_df = pd.read_json(repos_df_path, orient="records", lines=True)
        python_code_df = pd.read_feather(selected_python_code_path)
        sampled_repos_df = Code2DocSteps.create_repos_sample_df(
            repos_df,
            python_code_df,
            n_repos_per_task,
            min_task_size,
            max_task_size,
            max_random_baseline_score,
        )
        sampled_repos_df.to_json(output_path, orient="records", lines=True)
        logging.info(f"Sampled repos saved to {output_path}")

    @staticmethod
    def generate_code2doc_readmes_df(
        python_code_df,
        sampled_repos_df,
        files_per_repo=10,
        text_column="selected_code",
        lm_model_name="codellama",
        lm_base_url="http://localhost:11434",
    ):
        python_code_df = python_code_df[
            python_code_df["repo_name"].isin(sampled_repos_df["repo"])
        ]
        n_code_files = len(python_code_df)
        python_code_df = python_code_df.dropna(subset=["selected_code"])
        logging.info(f"Generating readmes for {len(sampled_repos_df)} repos.")
        logging.info(f"Dropped {n_code_files - len(python_code_df)} empty code files.")
        assert len(python_code_df["repo_name"].unique()) == len(
            sampled_repos_df["repo"]
        ), "Some repos are missing code files"
        logging.info(f"Using {len(python_code_df)} code files")
        avg_files_per_repo = len(python_code_df) / len(sampled_repos_df)
        logging.info(f"{round(avg_files_per_repo, 2)} files per repo on average")
        logging.info(f"Using {files_per_repo} files per repo")
        lm = dspy.OllamaLocal(model=lm_model_name, base_url=lm_base_url)
        dspy.configure(lm=lm)
        return run_code2doc_on_files_df(
            python_code_df,
            files_per_repo,
            text_column,
        )

    @staticmethod
    def generate_code2doc_readmes(
        sampled_repos_df_path,
        python_code_df_path,
        out_path,
        lm_model_name="codellama",
        lm_base_url="http://localhost:11434",
        files_per_repo=10,
    ):
        logging.info(f"Generating code2doc readmes using {lm_model_name}")
        python_code_df = pd.read_feather(python_code_df_path)
        sampled_repos_df = pd.read_json(
            sampled_repos_df_path, orient="records", lines=True
        )
        generated_readme_df = Code2DocSteps.generate_code2doc_readmes_df(
            python_code_df,
            sampled_repos_df,
            files_per_repo,
        )
        generated_readme_df.to_json(out_path, orient="records", lines=True)
        logging.info(f"Generated readmes saved to {out_path}")


# ploomber


def code2doc_prepare_data_pb(repos_df_path, python_code_path, product):
    Code2DocSteps.prepare_data(
        repos_df_path,
        python_code_path,
        product["repos_df_path"],
        product["selected_python_code_path"],
    )


def create_repos_sample_pb(
    upstream,
    product,
    n_repos_per_task,
    min_task_size,
    max_task_size,
    max_random_baseline_score,
):
    Code2DocSteps.create_repos_sample(
        str(upstream["code2doc.prepare_data"]["repos_df_path"]),
        str(upstream["code2doc.prepare_data"]["selected_python_code_path"]),
        product["sampled_repos"],
        n_repos_per_task,
        min_task_size,
        max_task_size,
        max_random_baseline_score,
    )


def generate_code2doc_readmes_pb(
    upstream,
    product,
    lm_model_name,
    lm_base_url,
    files_per_repo,
):
    logging.info(
        f"Generating readmes with code2doc using {lm_model_name}, using maximum of {files_per_repo} files per repo"
    )
    Code2DocSteps.generate_code2doc_readmes(
        str(upstream["code2doc.create_repo_sample"]["sampled_repos"]),
        str(upstream["code2doc.prepare_data"]["selected_python_code_path"]),
        str(product),
        lm_model_name,
        lm_base_url,
        files_per_repo=10,
    )
