import numpy as np
import pytest

from npl_portfolio.ml.calibration_evaluator import CalibrationEvaluator


def test_perfect_predictions():
    evaluator = CalibrationEvaluator(n_bins=10)

    result = evaluator.evaluate(
        [0, 1, 0, 1],
        [0.0, 1.0, 0.0, 1.0],
    )

    assert result.brier_score == pytest.approx(0.0)
    assert result.log_loss == pytest.approx(0.0, abs=1e-12)


def test_constant_probability():
    evaluator = CalibrationEvaluator(n_bins=10)

    result = evaluator.evaluate(
        [0, 1, 0, 1],
        [0.5, 0.5, 0.5, 0.5],
    )

    assert result.brier_score == pytest.approx(0.25)
    assert result.observed_prevalence == pytest.approx(0.5)
    assert result.mean_predicted_probability == pytest.approx(0.5)


def test_bin_counts_sum_to_total():
    evaluator = CalibrationEvaluator(n_bins=5)

    result = evaluator.evaluate(
        [0, 1, 0, 1, 1],
        [0.1, 0.3, 0.5, 0.7, 0.9],
    )

    assert sum(item["count"] for item in result.bins) == 5
    assert sum(item["positive_cases"] for item in result.bins) == 3


def test_probability_one_in_last_bin():
    evaluator = CalibrationEvaluator(n_bins=10)

    result = evaluator.evaluate([0, 1], [0.0, 1.0])

    last_bin = max(result.bins, key=lambda item: item["bin_index"])

    assert last_bin["bin_index"] == 9
    assert last_bin["positive_cases"] == 1


@pytest.mark.parametrize(
    "y,scores",
    [
        ([], []),
        ([0, 1], [0.5]),
        ([0, 2], [0.2, 0.8]),
        ([0, 1], [float("nan"), 0.8]),
        ([0, 1], [-0.1, 0.8]),
        ([0, 1], [0.2, 1.1]),
    ],
)
def test_invalid_inputs(y, scores):
    evaluator = CalibrationEvaluator()

    with pytest.raises(ValueError):
        evaluator.evaluate(y, scores)


@pytest.mark.parametrize("n_bins", [0, 1, 1.5, True])
def test_invalid_bins(n_bins):
    with pytest.raises(ValueError):
        CalibrationEvaluator(n_bins=n_bins)
