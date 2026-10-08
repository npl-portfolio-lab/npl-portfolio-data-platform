"""Calibradores reutilizables para probabilidades binarias."""

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted


class ProbabilityCalibrator:
    def __init__(self, method="sigmoid"):
        if method not in ("sigmoid", "isotonic"):
            raise ValueError("Método no soportado")

        self.method = method
        self.model = None

    @staticmethod
    def _validate_probabilities(probabilities):
        scores = np.asarray(probabilities, dtype=float)

        if scores.ndim != 1 or len(scores) == 0:
            raise ValueError("Probabilidades deben ser 1D y no vacías")

        if not np.isfinite(scores).all():
            raise ValueError("Probabilidades no finitas")

        if np.any((scores < 0) | (scores > 1)):
            raise ValueError("Probabilidades fuera de [0,1]")

        return scores

    def fit(self, probabilities, y_true):
        scores = self._validate_probabilities(probabilities)
        y = np.asarray(y_true)

        if y.ndim != 1 or len(y) != len(scores):
            raise ValueError("Dimensiones incompatibles")

        if not np.isin(y, [0, 1]).all():
            raise ValueError("Etiquetas no binarias")

        if len(np.unique(y)) != 2:
            raise ValueError("Se requieren ambas clases")

        if self.method == "sigmoid":
            self.model = LogisticRegression(
                solver="lbfgs",
                random_state=42,
                max_iter=1000,
            )
            self.model.fit(scores.reshape(-1, 1), y)

        else:
            self.model = IsotonicRegression(
                y_min=0.0,
                y_max=1.0,
                out_of_bounds="clip",
            )
            self.model.fit(scores, y)

        return self

    def predict_proba(self, probabilities):
        if self.model is None:
            raise ValueError("Calibrador no entrenado")

        scores = self._validate_probabilities(probabilities)

        if self.method == "sigmoid":
            check_is_fitted(self.model)
            calibrated = self.model.predict_proba(
                scores.reshape(-1, 1)
            )[:, 1]
        else:
            check_is_fitted(self.model)
            calibrated = self.model.predict(scores)

        return np.clip(
            np.asarray(calibrated, dtype=float),
            0.0,
            1.0,
        )
