from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class EvaluationResult:
    threshold: float
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    tn: int
    fp: int
    fn: int
    tp: int


class ModelEvaluator:
    """
    Evaluación reproducible de modelos binarios.

    Recibe probabilidades ya calculadas.
    No entrena modelos ni modifica datasets.
    """

    def __init__(
        self,
        threshold: float = 0.50,
    ) -> None:
        if not 0 < threshold < 1:
            raise ValueError(
                "threshold debe estar entre 0 y 1."
            )

        self.threshold = threshold

    def evaluate(
        self,
        y_true: np.ndarray,
        probabilities: np.ndarray,
    ) -> EvaluationResult:

        y_true = np.asarray(y_true)
        probabilities = np.asarray(probabilities)

        if y_true.ndim != 1 or probabilities.ndim != 1:
            raise ValueError(
                "Las entradas deben ser vectores 1D."
            )

        if len(y_true) != len(probabilities):
            raise ValueError(
                "TARGET y probabilidades tienen "
                "diferente longitud."
            )

        if len(y_true) == 0:
            raise ValueError("Dataset vacío.")

        if not np.isin(y_true, [0, 1]).all():
            raise ValueError(
                "TARGET debe contener solamente 0 y 1."
            )

        if len(np.unique(y_true)) != 2:
            raise ValueError(
                "Se requieren ambas clases para ROC-AUC."
            )

        if not np.isfinite(probabilities).all():
            raise ValueError(
                "Probabilidades con NaN o infinito."
            )

        if (
            (probabilities < 0)
            | (probabilities > 1)
        ).any():
            raise ValueError(
                "Las probabilidades deben estar entre 0 y 1."
            )

        predictions = (
            probabilities >= self.threshold
        ).astype(np.int8)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1],
        ).ravel()

        return EvaluationResult(
            threshold=self.threshold,
            roc_auc=float(
                roc_auc_score(y_true, probabilities)
            ),
            pr_auc=float(
                average_precision_score(
                    y_true,
                    probabilities,
                )
            ),
            precision=float(
                precision_score(
                    y_true,
                    predictions,
                    zero_division=0,
                )
            ),
            recall=float(
                recall_score(
                    y_true,
                    predictions,
                    zero_division=0,
                )
            ),
            f1=float(
                f1_score(
                    y_true,
                    predictions,
                    zero_division=0,
                )
            ),
            tn=int(tn),
            fp=int(fp),
            fn=int(fn),
            tp=int(tp),
        )
