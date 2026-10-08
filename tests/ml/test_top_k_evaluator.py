import numpy as np
import pytest

from npl_portfolio.ml.top_k_evaluator import TopKEvaluator


@pytest.fixture
def evaluator():
    return TopKEvaluator()


def test_top_k_metrics(evaluator):
    y = np.array([1, 0, 1, 0, 1])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])

    result = evaluator.evaluate(y, scores, (2,))[0]

    assert result.true_positives == 1
    assert result.precision_at_k == pytest.approx(0.5)
    assert result.recall_at_k == pytest.approx(1 / 3)
    assert result.lift_at_k == pytest.approx(0.5 / 0.6)


def test_perfect_ranking(evaluator):
    y = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.2, 0.1])

    result = evaluator.evaluate(y, scores, (2,))[0]

    assert result.true_positives == 2
    assert result.precision_at_k == 1.0
    assert result.recall_at_k == 1.0
    assert result.lift_at_k == 2.0


def test_all_clients_selected(evaluator):
    y = np.array([1, 0, 1, 0])
    scores = np.array([0.8, 0.7, 0.6, 0.5])

    result = evaluator.evaluate(y, scores, (4,))[0]

    assert result.true_positives == 2
    assert result.recall_at_k == 1.0
    assert result.lift_at_k == 1.0


def test_ties_are_deterministic(evaluator):
    y = np.array([1, 0, 1])
    scores = np.array([0.9, 0.9, 0.1])

    first = evaluator.evaluate(y, scores, (1,))[0]
    second = evaluator.evaluate(y, scores, (1,))[0]

    assert first == second
    assert first.true_positives == 1


@pytest.mark.parametrize(
    "y,scores,k",
    [
        ([0, 0], [0.8, 0.2], (1,)),
        ([1, 0], [0.8], (1,)),
        ([1, 0], [float("nan"), 0.2], (1,)),
        ([1, 0], [1.2, 0.2], (1,)),
        ([1, 0], [0.8, 0.2], (0,)),
        ([1, 0], [0.8, 0.2], (3,)),
        ([1, 0], [0.8, 0.2], (1.5,)),
        ([1, 0], [0.8, 0.2], ()),
    ],
)
def test_invalid_inputs(evaluator, y, scores, k):
    with pytest.raises(ValueError):
        evaluator.evaluate(y, scores, k)
