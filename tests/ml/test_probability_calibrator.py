import numpy as np
import pytest

from npl_portfolio.ml.probability_calibrator import (
    ProbabilityCalibrator,
)


@pytest.fixture
def sample():
    probabilities = np.array([
        0.05, 0.10, 0.20, 0.30, 0.40,
        0.50, 0.60, 0.70, 0.80, 0.95,
    ])
    y = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])
    return probabilities, y


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_calibrator_outputs_valid_probabilities(sample, method):
    probabilities, y = sample

    calibrator = ProbabilityCalibrator(method)
    calibrator.fit(probabilities, y)

    result = calibrator.predict_proba(probabilities)

    assert len(result) == len(probabilities)
    assert np.isfinite(result).all()
    assert ((result >= 0) & (result <= 1)).all()


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_calibrator_is_monotonic(sample, method):
    probabilities, y = sample

    calibrator = ProbabilityCalibrator(method)
    calibrator.fit(probabilities, y)

    result = calibrator.predict_proba(probabilities)

    assert np.all(np.diff(result) >= -1e-12)


def test_sigmoid_reproducibility(sample):
    probabilities, y = sample

    first = ProbabilityCalibrator("sigmoid").fit(
        probabilities, y
    )
    second = ProbabilityCalibrator("sigmoid").fit(
        probabilities, y
    )

    np.testing.assert_allclose(
        first.predict_proba(probabilities),
        second.predict_proba(probabilities),
    )


def test_predict_before_fit():
    calibrator = ProbabilityCalibrator("sigmoid")

    with pytest.raises(ValueError):
        calibrator.predict_proba([0.1, 0.2])


def test_invalid_method():
    with pytest.raises(ValueError):
        ProbabilityCalibrator("unknown")


@pytest.mark.parametrize(
    "probabilities,y",
    [
        ([0.1, 0.2], [0, 0]),
        ([0.1, 0.2], [0, 2]),
        ([0.1, 0.2], [0]),
        ([float("nan"), 0.2], [0, 1]),
        ([-0.1, 0.2], [0, 1]),
        ([0.1, 1.2], [0, 1]),
        ([], []),
    ],
)
def test_invalid_fit(probabilities, y):
    calibrator = ProbabilityCalibrator("sigmoid")

    with pytest.raises(ValueError):
        calibrator.fit(probabilities, y)


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_invalid_predict_probabilities(sample, method):
    probabilities, y = sample

    calibrator = ProbabilityCalibrator(method).fit(
        probabilities, y
    )

    with pytest.raises(ValueError):
        calibrator.predict_proba([float("inf")])
