import numpy as np
import pytest

from npl_portfolio.ml.model_evaluator import (
    ModelEvaluator,
)


def test_model_evaluator() -> None:
    y_true = np.array([0, 0, 1, 1])

    probabilities = np.array([
        0.10,
        0.60,
        0.40,
        0.90,
    ])

    evaluator = ModelEvaluator(
        threshold=0.50
    )

    result = evaluator.evaluate(
        y_true,
        probabilities,
    )

    assert result.tn == 1
    assert result.fp == 1
    assert result.fn == 1
    assert result.tp == 1

    assert result.precision == pytest.approx(0.5)
    assert result.recall == pytest.approx(0.5)
    assert result.f1 == pytest.approx(0.5)

    assert result.roc_auc == pytest.approx(0.75)
    assert result.pr_auc == pytest.approx(
        5 / 6
    )


def test_model_evaluator_rejects_invalid_probabilities() -> None:
    evaluator = ModelEvaluator()

    with pytest.raises(ValueError):
        evaluator.evaluate(
            np.array([0, 1]),
            np.array([0.2, np.nan]),
        )
