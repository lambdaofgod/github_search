import pytest
from github_search.lms.code2documentation import Code2Documentation, Prompts, RepoFileSummaryProvider
import dspy
from unittest.mock import patch, MagicMock


@pytest.fixture
def mock_dspy_settings():
    with patch("dspy.settings") as mock_settings:
        mock_settings.lm = MagicMock()
        mock_settings.lm.kwargs = {}
        yield mock_settings


from github_search.lms.repo_file_summary_provider import RepoMapProvider

def test_code2doc_basic(mock_dspy_settings):
    # Mock RepoMapProvider
    repo_maps = {
        "test_repo": "f1.py\nf2.txt\nf3.py\n",
    }
    repo_map_provider = RepoMapProvider(repo_maps=repo_maps)

    # Initialize Code2Documentation
    code2doc = Code2Documentation(repo_file_summary_provider=repo_map_provider)

    # Mock dspy.Predict and dspy.ChainOfThought
    code2doc.summarize_files = lambda **kwargs: {
        "answer": "This file imports the os module."
    }
    code2doc.summarize_repo = lambda **kwargs: {
        "answer": "This repository contains Python utility functions."
    }

    # Run the forward method
    result = code2doc("test_repo")

    # Assert the results
    assert result.repo_name == "test_repo"
    assert result.answer == "This repository contains Python utility functions."
    assert result.context_history == "This file imports the os module."
    assert result.filenames == ["f1.py"]
    assert result.n_files == 1
