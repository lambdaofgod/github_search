from contextlib import contextmanager
import pandas as pd
import dspy
import re
import tqdm
from github_search.lms.utils import enable_phoenix_tracing
from github_search.lms.repo_file_summary_provider import (
    DataFrameRepoFileSummaryProvider,
    RepoFileSummaryProvider,
    DataFrameTextProvider,
)
import logging


class Code2DocGenerationException(Exception):

    def __init__(self, message):
        self.message = message
        super().__init__(message)


@contextmanager
def override_lm_params(**kwargs):
    lm = dspy.settings.lm
    old_kwargs = {param_name: lm.kwargs[param_name] for param_name in kwargs.keys()}
    try:
        for param_name, param_value in kwargs.items():
            lm.kwargs[param_name] = param_value
        yield
    finally:
        for param_name, param_value in old_kwargs.items():
            lm.kwargs[param_name] = param_value


class MultiFileSummary(dspy.Signature):
    question = dspy.InputField()
    context = dspy.InputField(desc="Python code")
    answer = dspy.OutputField(desc="Summary of the code given guiding question")


class RepoSummary(dspy.Signature):
    context = dspy.InputField(desc="Python file summaries")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="Repository summary")


class Prompts:
    file_summary_question_template = """
    given the code extracted from Python files of {} repository,
    separated with ``` describe what each file implements in 3 sentences.
    Focus on machine learning models and data."""

    repo_summary_question_template = """
    Using summaries of '{}' files from Context, write repository README.
    Focus on the functionalities and features.
    There is no need to describe the dependencies and setup.
    The README should provide answers to the following questions:
    - what machine learning problem does this repository tackle?
    - what kind of data does it use?
    Base your answer only on the information from context.
    """.strip()


class DirectRepoSummary(dspy.Signature):
    context = dspy.InputField(desc="Python code from repository files")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="Repository README documentation")


class Code2DocumentationFlat(dspy.Module):
    def __init__(
        self,
        repo_file_summary_provider: RepoFileSummaryProvider,
        repo_summary_question_template=Prompts.repo_summary_question_template,
        verbose=True,
    ):
        super().__init__()
        self.repo_file_summary_provider = repo_file_summary_provider
        self.summarize_repo = dspy.ChainOfThought(DirectRepoSummary)
        self.repo_summary_question_template = repo_summary_question_template

    def _create_multi_file_input(self, paths, code_file_contents):
        return "\n\n".join(
            [
                self._create_single_file_part(path, code)
                for path, code in zip(paths, code_file_contents)
            ]
        )

    def _create_single_file_part(self, path, code):
        return f"file {path}\n```\n{code}\n```"

    def forward(
        self,
        repo_name,
        repo_summary_kwargs={"num_predict": 256},
        lms=None,
    ):
        try:
            filenames = self.repo_file_summary_provider.get_filenames(repo_name)
            code_context = self.repo_file_summary_provider.extract_summary(repo_name)
            if isinstance(filenames, pd.Series):
                filenames = filenames.tolist()

            for key in repo_summary_kwargs.keys():
                dspy.settings.lm.kwargs[key] = repo_summary_kwargs[key]

            repo_summary = self.summarize_repo(
                question=self.repo_summary_question_template.format(repo_name),
                context=code_context,
            )

            return dspy.Prediction(
                **repo_summary,
                repo_name=repo_name,
                filenames=filenames,
                n_files=len(filenames),
            )
        except Exception as e:
            raise Code2DocGenerationException(f"Error processing {repo_name}: {e}")


class Code2Documentation(dspy.Module):
    def __init__(
        self,
        repo_file_summary_provider: RepoFileSummaryProvider,
        repo_summary_question_template=Prompts.repo_summary_question_template,
        file_summary_question_template=Prompts.file_summary_question_template,
        verbose=True,
    ):
        super().__init__()
        self.repo_file_summary_provider = repo_file_summary_provider
        self.summarize_files = dspy.Predict(MultiFileSummary)
        self.summarize_repo = dspy.ChainOfThought(RepoSummary)
        self.file_summary_question_template = file_summary_question_template
        self.repo_summary_question_template = repo_summary_question_template

    def _create_multi_file_input(self, paths, code_file_contents):
        return "\n\n".join(
            [
                self._create_single_file_part(path, code)
                for path, code in zip(paths, code_file_contents)
            ]
        )

    def _create_single_file_part(self, path, code):
        return f"file {path}\n```\n{code}\n```"

    def _summarize_files(self, repo_name):
        filenames = self.repo_file_summary_provider.get_filenames(repo_name)
        summary = self.repo_file_summary_provider.extract_summary(repo_name)
        if isinstance(filenames, pd.Series):
            filenames = filenames.tolist()
        return filenames, self.summarize_files(
            question=self.file_summary_question_template.format(repo_name),
            context=summary,
        )

    def forward(
        self,
        repo_name,
        file_summary_kwargs={"num_predict": 1024},
        repo_summary_kwargs={"num_predict": 256},
        lms=None,
    ):
        try:
            if lms is not None:
                return self.forward_multilm(repo_name, lms)
            else:
                for key in file_summary_kwargs.keys():
                    dspy.settings.lm.kwargs[key] = file_summary_kwargs[key]
                filenames, summary_result = self._summarize_files(repo_name)
                summaries = summary_result["answer"]
                for key in repo_summary_kwargs.keys():
                    dspy.settings.lm.kwargs[key] = repo_summary_kwargs[key]
                repo_summary = self.summarize_repo(
                    question=self.repo_summary_question_template.format(repo_name),
                    context=summaries,
                )

                return dspy.Prediction(
                    **repo_summary,
                    repo_name=repo_name,
                    context_history=summaries,
                    filenames=filenames,
                    n_files=len(filenames),
                )
        except Exception as e:
            raise Code2DocGenerationException(f"Error processing {repo_name}: {e}")

    def forward_multilm(self, repo_name, lms):
        [small_lm, bigger_lm] = lms
        dspy.configure(lm=small_lm)
        filenames, summary_result = self._summarize_files(repo_name)
        summaries = summary_result["answer"]

        dspy.configure(lm=bigger_lm)
        repo_summary = self.summarize_repo(
            question=self.repo_summary_question_template.format(repo_name),
            context=summaries,
        )

        return dspy.Prediction(
            **repo_summary,
            repo_name=repo_name,
            context_history=summaries,
            filenames=filenames,
            n_files=len(filenames),
        )


def run_code2doc_on_files_df(
    python_files_df, files_per_repo, code_col="selected_code", use_phoenix=True, use_flat=False
):
    repo_file_summary_provider = DataFrameRepoFileSummaryProvider(
        python_files_df, files_per_repo, code_col
    )

    if use_flat:
        code2doc = Code2DocumentationFlat(repo_file_summary_provider=repo_file_summary_provider)
    else:
        code2doc = Code2Documentation(repo_file_summary_provider=repo_file_summary_provider)
    
    code2doc_answers = []
    if use_phoenix:
        enable_phoenix_tracing()
    for repo_name in tqdm.tqdm(python_files_df["repo_name"].unique()):
        try:
            code2doc_answers.append(dict(code2doc(repo_name)))
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as e:
            logging.error(f"Error processing {repo_name}: {e}")

    return pd.DataFrame.from_records(code2doc_answers)


def run_code2doc_on_df(repo_df, code_col, name_col, use_phoenix=True, use_flat=False):
    repo_file_summary_provider = DataFrameTextProvider(
        df=repo_df,
        text_col=code_col,
        name_col=name_col,
    )

    if use_flat:
        code2doc = Code2DocumentationFlat(repo_file_summary_provider=repo_file_summary_provider)
    else:
        code2doc = Code2Documentation(repo_file_summary_provider=repo_file_summary_provider)
    
    code2doc_answers = []
    if use_phoenix:
        enable_phoenix_tracing()
    for repo_name in tqdm.tqdm(repo_df[name_col].unique()):
        try:
            code2doc_answers.append(dict(code2doc(repo_name)))
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Code2DocGenerationException as e:
            logging.error(e.message)
            pass

    return pd.DataFrame.from_records(code2doc_answers)


