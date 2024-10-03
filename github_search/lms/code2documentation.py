from contextlib import contextmanager
import pandas as pd
import dspy
import re
import tqdm
from github_search.lms.utils import enable_phoenix_tracing
import logging


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


class RepoFileSummaryProvider(abc.ABC):
    def extract_summary(self, repo_name) -> str:
        pass

    def get_filenames(self, repo_name) -> List[str]:
        pass


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


def run_code2doc(
    python_files_df, files_per_repo, code_col="selected_code", use_phoenix=True
):
    class DataFrameRepoFileSummaryProvider(RepoFileSummaryProvider):
        def __init__(self, df, files_per_repo, code_col):
            self.df = df
            self.files_per_repo = files_per_repo
            self.code_col = code_col

        def get_filenames(self, repo_name):
            return self.df[self.df["repo_name"] == repo_name].iloc[:self.files_per_repo]["path"]

        def extract_summary(self, repo_name):
            selected_python_files = self.df[self.df["repo_name"] == repo_name].iloc[:self.files_per_repo]
            return "\n\n".join([
                f"file {path}\n```\n{code}\n```"
                for path, code in zip(selected_python_files["path"], selected_python_files[self.code_col])
            ])

    repo_file_summary_provider = DataFrameRepoFileSummaryProvider(python_files_df, files_per_repo, code_col)
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
