import numpy as np
import pytest
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MaxAbsScaler

from npl_portfolio.ml.model_evaluator import ModelEvaluator


@pytest.fixture
def sample_data():
    X_train = sparse.csr_matrix(
        np.array([
            [1.0, 0.0],
            [2.0, 1.0],
            [3.0, 0.0],
            [4.0, 1.0],
            [5.0, 0.0],
            [6.0, 1.0],
        ])
    )
    y_train = np.array([0, 0, 0, 1, 1, 1])

    X_validation = sparse.csr_matrix(
        np.array([[2.0, 0.0], [5.0, 1.0]])
    )
    y_validation = np.array([0, 1])

    return X_train, y_train, X_validation, y_validation


def test_scaler_fitted_only_on_train(sample_data):
    X_train, _, X_validation, _ = sample_data

    scaler = MaxAbsScaler()
    scaler.fit(X_train)

    np.testing.assert_allclose(
        scaler.max_abs_,
        [6.0, 1.0],
    )

    transformed = scaler.transform(X_validation)

    assert transformed.shape == X_validation.shape


@pytest.mark.parametrize("weight", [None, "balanced"])
def test_model_training_and_probabilities(sample_data, weight):
    X_train, y_train, X_validation, _ = sample_data

    scaler = MaxAbsScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(
        solver="liblinear",
        max_iter=200,
        tol=1e-3,
        random_state=42,
        class_weight=weight,
    )

    model.fit(X_scaled, y_train)

    probabilities = model.predict_proba(
        scaler.transform(X_validation)
    )[:, 1]

    assert probabilities.shape == (2,)
    assert np.isfinite(probabilities).all()
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert model.class_weight == weight


def test_evaluator_confusion_matrix_consistency():
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.8, 0.7, 0.2])

    result = ModelEvaluator(threshold=0.5).evaluate(
        y_true, probabilities
    )

    assert result.tn == 1
    assert result.fp == 1
    assert result.fn == 1
    assert result.tp == 1

    assert result.tn + result.fp + result.fn + result.tp == 4
    assert result.precision == pytest.approx(0.5)
    assert result.recall == pytest.approx(0.5)
