"""Bootstrap pareado estratificado para comparar clasificadores."""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass(frozen=True)
class BootstrapResult:
    metric: str
    observed_difference: float
    mean_difference: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    bootstrap_iterations: int
    interval_includes_zero: bool


class PairedBootstrap:
    def __init__(
        self,
        iterations=1000,
        confidence_level=0.95,
        random_state=42,
    ):
        if isinstance(iterations, bool) or not isinstance(
            iterations, (int, np.integer)
        ) or iterations < 2:
            raise ValueError("iterations debe ser entero >= 2")

        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level debe estar entre 0 y 1")

        self.iterations = int(iterations)
        self.confidence_level = confidence_level
        self.random_state = random_state

    @staticmethod
    def _metric(y, probabilities, metric, k=None):
        if metric == "roc_auc":
            return float(roc_auc_score(y, probabilities))

        if metric == "pr_auc":
            return float(average_precision_score(y, probabilities))

        if metric == "precision_at_k":
            if k is None or not 1 <= k <= len(y):
                raise ValueError("K inválido")

            ranking = np.argsort(-probabilities, kind="stable")
            return float(np.mean(y[ranking[:k]]))

        raise ValueError(f"Métrica no soportada: {metric}")

    def compare(
        self,
        y_true,
        probabilities_a,
        probabilities_b,
        metric,
        k=None,
    ):
        y = np.asarray(y_true)
        a = np.asarray(probabilities_a)
        b = np.asarray(probabilities_b)

        if y.ndim != 1 or a.ndim != 1 or b.ndim != 1:
            raise ValueError("Entradas deben ser unidimensionales")

        if len(y) == 0 or len(y) != len(a) or len(y) != len(b):
            raise ValueError("Dimensiones incompatibles")

        if not np.isin(y, [0, 1]).all():
            raise ValueError("Etiquetas no binarias")

        if len(np.unique(y)) != 2:
            raise ValueError("Se requieren ambas clases")

        for probabilities in (a, b):
            if not np.isfinite(probabilities).all():
                raise ValueError("Probabilidades no finitas")

            if np.any((probabilities < 0) | (probabilities > 1)):
                raise ValueError("Probabilidades fuera de [0,1]")

        if metric == "precision_at_k":
            if isinstance(k, (bool, np.bool_)) or not isinstance(
                k, (int, np.integer)
            ) or not 1 <= k <= len(y):
                raise ValueError("K inválido")

        observed = (
            self._metric(y, b, metric, k)
            - self._metric(y, a, metric, k)
        )

        rng = np.random.default_rng(self.random_state)

        negative_indices = np.flatnonzero(y == 0)
        positive_indices = np.flatnonzero(y == 1)

        differences = np.empty(self.iterations, dtype=float)

        for iteration in range(self.iterations):
            negatives = rng.choice(
                negative_indices,
                size=len(negative_indices),
                replace=True,
            )

            positives = rng.choice(
                positive_indices,
                size=len(positive_indices),
                replace=True,
            )

            indices = np.concatenate((negatives, positives))

            # Mezcla común para ambos modelos.
            rng.shuffle(indices)

            sample_y = y[indices]

            score_a = self._metric(
                sample_y,
                a[indices],
                metric,
                k,
            )

            score_b = self._metric(
                sample_y,
                b[indices],
                metric,
                k,
            )

            differences[iteration] = score_b - score_a

        alpha = 1 - self.confidence_level

        lower, upper = np.quantile(
            differences,
            [alpha / 2, 1 - alpha / 2],
        )

        return BootstrapResult(
            metric=metric if k is None else f"{metric}_{k}",
            observed_difference=float(observed),
            mean_difference=float(np.mean(differences)),
            ci_lower=float(lower),
            ci_upper=float(upper),
            confidence_level=float(self.confidence_level),
            bootstrap_iterations=self.iterations,
            interval_includes_zero=bool(lower <= 0 <= upper),
        )
