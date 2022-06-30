import numpy as np
from github_search import utils
from dataclasses import dataclass
import pytest


test_inputs = [
    dict(
        labels=np.array([[1, 0], [1, 0]]),
        scores=np.array([[0, 1], [0.0, 1.0]]),
        expected_score=0,
    ),
    dict(
        labels=np.array([[0, 1], [0, 1]]),
        scores=np.array([[0, 0.5], [0.0, 1.0]]),
        expected_score=1,
    ),
    dict(
        labels=np.array([[1, 0], [0, 1]]),
        scores=np.array([[0, 0.5], [0.5, 1.0]]),
        expected_score=1/2,
    ),
    dict(
        labels=np.array([[1, 0, 0], [0, 1, 1]]),
        scores=np.array([[0.5, 0.0, 0.0], [0.5, 1.0, 1.0]]),
        expected_score=1,
    ),
    dict(
        labels=np.array([[1, 0, 0], [0, 1, 1]]),
        scores=np.array([[0, 0.5, 0.0], [0.5, 1.0, 1.0]]),
        expected_score=2/3,
    ),
]

@pytest.mark.parametrize("input", test_inputs)
def test_multilabel_samplewise_topk_accuracy(input):
    result = utils.get_multilabel_samplewise_topk_accuracy(
        input['labels'], input['scores']
    )
    expected_result = input['expected_score']
    assert np.isclose(result, expected_result), f"{result} != {expected_result}"


def test_accuracy_from_labels():
    labels = [0, 1]
    scores = np.array([[0.5, 0], [0, 1]])
    assert np.isclose(utils.get_accuracy_from_scores(labels, scores), 1)
