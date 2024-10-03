import pytest
from github_search.lms.code2documentation import Code2Documentation, Prompts
import dspy
from unittest.mock import patch, MagicMock


@pytest.fixture
def mock_dspy_settings():
    with patch("dspy.settings") as mock_settings:
        mock_settings.lm = MagicMock()
        mock_settings.lm.kwargs = {}
        yield mock_settings


def test_code2doc_basic(mock_dspy_settings):
    # Mock RepoFileSummaryProvider
    class MockRepoFileSummaryProvider(RepoFileSummaryProvider):
        def get_filenames(self, repo_name):
            return ["f1.py"]

        def extract_summary(self, repo_name):
            return "file f1.py\n```\nimport os\n```"

    # Initialize Code2Documentation
    code2doc = Code2Documentation(repo_file_summary_provider=MockRepoFileSummaryProvider())

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
