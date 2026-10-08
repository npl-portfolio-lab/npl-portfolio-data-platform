import numpy as np
import pytest

from npl_portfolio.ml.paired_bootstrap import PairedBootstrap


@pytest.fixture
def sample():
    y = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    a = np.array([0.9, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    b = np.array([0.9, 0.8, 0.2, 0.1, 0.7, 0.3, 0.6, 0.4])
    return y, a, b


def test_reproducibility(sample):
    y, a, b = sample
    bootstrap = PairedBootstrap(iterations=30, random_state=42)

    first = bootstrap.compare(y, a, b, "roc_auc")
    second = bootstrap.compare(y, a, b, "roc_auc")

    assert first == second


def test_identical_models_have_zero_difference(sample):
    y, a, _ = sample
    bootstrap = PairedBootstrap(iterations=30)

    result = bootstrap.compare(y, a, a, "pr_auc")

    assert result.observed_difference == 0
    assert result.ci_lower == 0
    assert result.ci_upper == 0
    assert result.interval_includes_zero


def test_better_model_has_positive_observed_difference(sample):
    y, a, b = sample
    bootstrap = PairedBootstrap(iterations=30)

    result = bootstrap.compare(y, a, b, "roc_auc")

    assert result.observed_difference > 0


@pytest.mark.parametrize(
    "metric,k",
    [
        ("roc_auc", None),
        ("pr_auc", None),
        ("precision_at_k", 3),
    ],
)
def test_supported_metrics(sample, metric, k):
    y, a, b = sample
    bootstrap = PairedBootstrap(iterations=20)

    result = bootstrap.compare(y, a, b, metric, k)

    assert np.isfinite(result.observed_difference)
    assert result.ci_lower <= result.ci_upper


@pytest.mark.parametrize(
    "y,a,b,metric,k",
    [
        ([0, 0], [0.2, 0.3], [0.3, 0.4], "roc_auc", None),
        ([1, 0], [0.2], [0.3, 0.4], "roc_auc", None),
        ([1, 0], [0.2, 0.3], [float("nan"), 0.4], "roc_auc", None),
        ([1, 0], [0.2, 0.3], [0.3, 1.2], "roc_auc", None),
        ([1, 0], [0.2, 0.3], [0.3, 0.4], "precision_at_k", 3),
        ([1, 0], [0.2, 0.3], [0.3, 0.4], "unknown", None),
    ],
)
def test_invalid_inputs(y, a, b, metric, k):
    bootstrap = PairedBootstrap(iterations=10)

    with pytest.raises(ValueError):
        bootstrap.compare(y, a, b, metric, k)


def test_invalid_iterations():
    with pytest.raises(ValueError):
        PairedBootstrap(iterations=1)


def test_invalid_confidence():
    with pytest.raises(ValueError):
        PairedBootstrap(confidence_level=1.0)
