import pytest
from github_search.lms.code2documentation import Code2Documentation, Prompts
import dspy

def test_code2doc_basic():
    # Mock fetch_code function
    def fetch_code_fn(repo_name):
        return (["f1.py"], ["import os"])

    # Initialize Code2Documentation
    code2doc = Code2Documentation(fetch_code_fn=fetch_code_fn)

    # Mock dspy.Predict and dspy.ChainOfThought
    code2doc.summarize_files = lambda **kwargs: {"answer": "This file imports the os module."}
    code2doc.summarize_repo = lambda **kwargs: {"answer": "This repository contains Python utility functions."}

    # Run the forward method
    result = code2doc("test_repo")

    # Assert the results
    assert result.repo_name == "test_repo"
    assert result.answer == "This repository contains Python utility functions."
    assert result.context_history == "This file imports the os module."
    assert result.filenames == ["f1.py"]
    assert result.n_files == 1

    # Test with custom templates
    custom_file_template = "Summarize this {} file:"
    custom_repo_template = "What does the {} repo do?"
    
    custom_code2doc = Code2Documentation(
        fetch_code_fn=fetch_code_fn,
        file_summary_question_template=custom_file_template,
        repo_summary_question_template=custom_repo_template
    )

    assert custom_code2doc.file_summary_question_template == custom_file_template
    assert custom_code2doc.repo_summary_question_template == custom_repo_template
