from contextlib import contextmanager
import pandas as pd
import dspy
import re
import tqdm
from github_search.lms.utils import enable_phoenix_tracing


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
    context = dspy.InputField(desc="Python code")
    question = dspy.InputField()
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


class Code2Documentation(dspy.Module):
    def __init__(
        self,
        fetch_code_fn,
        repo_summary_question_template=Prompts.repo_summary_question_template,
        file_summary_question_template=Prompts.file_summary_question_template,
        verbose=True,
    ):
        super().__init__()
        self.fetch_code = fetch_code_fn
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
        paths, code_file_contents = self.fetch_code(repo_name)
        file_summarization_input = self._create_multi_file_input(
            paths, code_file_contents
        )
        return self.summarize_files(
            question=self.file_summary_question_template.format(repo_name),
            context=file_summarization_input,
        )

    def forward(
        self,
        repo_name,
        file_summarizer_lm_kwargs={"num_predict": 1024},
        repo_summarizer_lm_kwargs={"num_ctx": 1024, "num_predict": 256},
    ):
        with override_lm_params(**file_summarizer_lm_kwargs):
            summaries = self._summarize_files(repo_name)["answer"]

        with override_lm_params(**repo_summarizer_lm_kwargs):
            repo_summary = self.summarize_repo(
                question=self.repo_summary_question_template.format(repo_name),
                context=summaries,
            )

        return dspy.Prediction(**repo_summary, context_history=summaries)


def run_code2doc(
    python_files_df, lm, files_per_repo, code_col="selected_code", use_phoenix=True
):
    def fetch_code(repo_name):
        selected_python_files = python_files_df[
            python_files_df["repo_name"] == repo_name
        ].iloc[:files_per_repo]
        return selected_python_files["path"], selected_python_files[code_col]

    dspy.configure(lm=lm)
    code2documentation = Code2Documentation(fetch_code_fn=fetch_code)
    code2doc_answers = []
    if use_phoenix:
        enable_phoenix_tracing()
    for repo_name in tqdm.tqdm(python_files_df["repo_name"].unique()):
        code2doc_answers.append(dict(code2documentation(repo_name)))

    return pd.DataFrame.from_records(code2doc_answers)
