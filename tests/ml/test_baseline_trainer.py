import numpy as np
from scipy import sparse

from npl_portfolio.ml.baseline_trainer import (
    BaselineTrainer,
)


def test_baseline_trainer() -> None:
    X_train = sparse.csr_matrix(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [2.0, 2.0],
        ],
        dtype=np.float32,
    )

    y_train = np.array(
        [
            0,
            0,
            0,
            1,
            1,
            1,
        ],
        dtype=np.int8,
    )

    X_validation = sparse.csr_matrix(
        [
            [0.0, 0.0],
            [0.5, 0.5],
            [1.5, 1.0],
            [2.0, 2.0],
        ],
        dtype=np.float32,
    )

    y_validation = np.array(
        [
            0,
            0,
            1,
            1,
        ],
        dtype=np.int8,
    )

    trainer = BaselineTrainer(
        random_state=42,
        threshold=0.50,
    )

    trainer.fit(
        X_train,
        y_train,
    )

    probabilities = trainer.predict_proba(
        X_validation
    )

    assert probabilities.shape == (4,)

    assert np.all(
        (probabilities >= 0)
        & (probabilities <= 1)
    )

    metrics = trainer.evaluate(
        X_validation,
        y_validation,
    )

    assert 0 <= metrics.roc_auc <= 1
    assert 0 <= metrics.pr_auc <= 1
    assert 0 <= metrics.precision <= 1
    assert 0 <= metrics.recall <= 1
    assert 0 <= metrics.f1 <= 1

    assert metrics.confusion_matrix.shape == (
        2,
        2,
    )

    assert (
        metrics.confusion_matrix.sum()
        == len(y_validation)
    )
