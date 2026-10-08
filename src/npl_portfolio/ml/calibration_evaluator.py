"""Evaluación probabilística y análisis de calibración."""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss


@dataclass(frozen=True)
class CalibrationResult:
    brier_score: float
    log_loss: float
    mean_predicted_probability: float
    observed_prevalence: float
    bins: list[dict]


class CalibrationEvaluator:
    def __init__(self, n_bins=10):
        if (
            isinstance(n_bins, bool)
            or not isinstance(n_bins, (int, np.integer))
            or n_bins < 2
        ):
            raise ValueError("n_bins debe ser entero >= 2")

        self.n_bins = int(n_bins)

    def evaluate(self, y_true, probabilities):
        y = np.asarray(y_true)
        scores = np.asarray(probabilities)

        if y.ndim != 1 or scores.ndim != 1:
            raise ValueError("Entradas deben ser unidimensionales")

        if len(y) == 0 or len(y) != len(scores):
            raise ValueError("Dimensiones incompatibles")

        if not np.isin(y, [0, 1]).all():
            raise ValueError("Etiquetas no binarias")

        if not np.isfinite(scores).all():
            raise ValueError("Probabilidades no finitas")

        if np.any((scores < 0) | (scores > 1)):
            raise ValueError("Probabilidades fuera de [0,1]")

        edges = np.linspace(0.0, 1.0, self.n_bins + 1)

        # El último intervalo incluye la probabilidad 1.0.
        indices = np.searchsorted(
            edges[1:-1],
            scores,
            side="right",
        )

        bins = []

        for index in range(self.n_bins):
            mask = indices == index
            count = int(np.sum(mask))

            if count == 0:
                continue

            bins.append({
                "bin_index": index,
                "lower_bound": float(edges[index]),
                "upper_bound": float(edges[index + 1]),
                "count": count,
                "mean_predicted_probability": float(
                    np.mean(scores[mask])
                ),
                "observed_positive_rate": float(
                    np.mean(y[mask])
                ),
                "positive_cases": int(np.sum(y[mask])),
            })

        return CalibrationResult(
            brier_score=float(brier_score_loss(y, scores)),
            log_loss=float(log_loss(y, scores, labels=[0, 1])),
            mean_predicted_probability=float(np.mean(scores)),
            observed_prevalence=float(np.mean(y)),
            bins=bins,
        )
