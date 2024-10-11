from typing import Annotated, List, Tuple

import pandas as pd
from github_search.ir.evaluator import (
    InformationRetrievalEvaluator,
    SearchDataFrameExtractor,
)
from github_search.pipelines.metrics_comparison import *
from github_search.pipelines.postprocessing import GenerationPostprocessor

from github_search.python_code import code_selection

try:
    from tgutil.configs import PromptConfig, load_config_from_dict
    from tgutil.prompting import ContextPromptInfo
    from tgutil.prompting_runner import DocumentExpander
except:
    logging.warning("tgutil not installed, using local imports")
from github_search.samplers import TaskSizeRepoSampler
import dspy
from github_search.lms.code2documentation import run_code2doc


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
        text_generation_config = load_config_from_dict(text_generation_config)
        prompt_config = PromptConfig(**prompt_config)
        parsed_prompt_infos = [ContextPromptInfo.parse_obj(pi) for pi in prompt_infos]
        raw_generated_texts_df, failures = DocumentExpander(
            text_generation_config=text_generation_config, prompts_config=prompt_config
        ).expand_documents(parsed_prompt_infos)
        assert (
            len(raw_generated_texts_df)
            == len(parsed_prompt_infos) * text_generation_config.n_generations
        ), "generating failed"
        raw_generated_texts_df = GenerationPostprocessor.convert_cols_to_dict(
            raw_generated_texts_df, ["prompt_info", "context_prompt_infos"]
        )
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
    def prepare_data(
        repos_df_path, python_code_path, repos_output_path, selected_python_code_path
    ):
        logging.info("loading repo data from %s", repos_df_path)
        repos_df = pd.read_json(repos_df_path)
        repos_df = repos_df[~repos_df["readme"].isna()]
        logging.info("loading python code data from %s", python_code_path)
        python_code_df = pd.read_parquet(python_code_path)
        repos_with_all_data_df = repos_df[
            repos_df["repo"].isin(python_code_df["repo_name"])
        ]
        repos_with_all_data_df.to_json(repos_output_path, orient="records", lines=True)
        # for some reason there are errors in parquet so we'll save it to feather
        code_selection.get_python_files_with_selected_code_df(
            python_code_df
        ).to_feather(selected_python_code_path)

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
        logging.basicConfig(level=logging.INFO)
        repos_df = pd.read_json(repos_df_path, orient="records", lines=True)
        python_code_df = pd.read_feather(selected_python_code_path)
        python_code_df = python_code_df.dropna(subset=["selected_code"])
        repos_df = repos_df[repos_df["repo"].isin(python_code_df["repo_name"])]
        repos_df["tasks"] = repos_df["tasks"].apply(ast.literal_eval)
        repo_sampler = TaskSizeRepoSampler(
            min_task_count=min_task_size,
            n_repos_per_task=n_repos_per_task,
            max_repos_per_task=max_task_size,
        )
        sampled_repos_df = repo_sampler.get_sampled_task_repos_df(
            repos_df, max_baseline_score=max_random_baseline_score
        )
        sampled_repos_df.to_json(output_path, orient="records", lines=True)

    @staticmethod
    def generate_code2doc_readmes(
        sampled_repos_df_path,
        python_code_df_path,
        out_path,
        lm_model_name="codellama",
        lm_base_url="http://localhost:11434",
        small_lm_base_url="http://localhost:11431",
        files_per_repo=10,
    ):
        python_code_df = pd.read_feather(python_code_df_path)
        sampled_repos_df = pd.read_json(
            sampled_repos_df_path, orient="records", lines=True
        )
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
        lm = dspy.HFClientVLLM(model="", port=8000, url="http://localhost")
        dspy.configure(lm=lm)
        generated_readme_df = run_code2doc(
            python_code_df,
            files_per_repo,
            "selected_code",
        )
        generated_readme_df.to_json(out_path, orient="records", lines=True)


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
