from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class BaselineMetrics:
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: np.ndarray


class BaselineTrainer:
    """
    Entrena y evalúa un baseline de clasificación binaria.

    El modelo se ajusta exclusivamente con TRAIN.
    VALIDATION únicamente se utiliza para evaluación.
    """

    def __init__(
        self,
        random_state: int = 42,
        threshold: float = 0.50,
    ) -> None:
        if not 0 < threshold < 1:
            raise ValueError(
                "threshold debe estar entre 0 y 1."
            )

        self.threshold = threshold

        self.model = LogisticRegression(
            solver="liblinear",
            max_iter=1000,
            random_state=random_state,
        )

    def fit(
        self,
        X: sparse.csr_matrix,
        y: np.ndarray,
    ) -> None:
        self.model.fit(
            X,
            y,
        )

    def predict_proba(
        self,
        X: sparse.csr_matrix,
    ) -> np.ndarray:
        return self.model.predict_proba(
            X
        )[:, 1]

    def evaluate(
        self,
        X: sparse.csr_matrix,
        y: np.ndarray,
    ) -> BaselineMetrics:
        probabilities = self.predict_proba(X)

        predictions = (
            probabilities >= self.threshold
        ).astype(np.int8)

        return BaselineMetrics(
            roc_auc=roc_auc_score(
                y,
                probabilities,
            ),
            pr_auc=average_precision_score(
                y,
                probabilities,
            ),
            precision=precision_score(
                y,
                predictions,
                zero_division=0,
            ),
            recall=recall_score(
                y,
                predictions,
                zero_division=0,
            ),
            f1=f1_score(
                y,
                predictions,
                zero_division=0,
            ),
            confusion_matrix=confusion_matrix(
                y,
                predictions,
                labels=[0, 1],
            ),
        )
