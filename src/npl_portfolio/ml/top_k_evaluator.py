"""Evaluación de priorización operativa Top-K."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TopKResult:
    k: int
    true_positives: int
    precision_at_k: float
    recall_at_k: float
    lift_at_k: float


class TopKEvaluator:
    def evaluate(
        self,
        y_true,
        probabilities,
        k_values=(1000, 3000, 5000),
    ) -> list[TopKResult]:
        y = np.asarray(y_true)
        scores = np.asarray(probabilities)

        if y.ndim != 1 or scores.ndim != 1:
            raise ValueError("Las entradas deben ser unidimensionales.")

        if len(y) == 0 or len(y) != len(scores):
            raise ValueError("Dimensiones incompatibles o datos vacíos.")

        if not np.isin(y, [0, 1]).all():
            raise ValueError("Las etiquetas deben ser binarias.")

        if not np.isfinite(scores).all():
            raise ValueError("Existen probabilidades no finitas.")

        if np.any((scores < 0) | (scores > 1)):
            raise ValueError("Las probabilidades deben estar entre 0 y 1.")

        total_positives = int(np.sum(y))
        if total_positives == 0:
            raise ValueError("No existen casos positivos para evaluar.")

        if not k_values:
            raise ValueError("Debe especificar al menos un valor K.")

        n = len(y)
        prevalence = total_positives / n

        # Ordenamiento determinista: ante empates, conserva
        # el orden original de los registros.
        ranking = np.argsort(-scores, kind="stable")
        ranked_targets = y[ranking]
        cumulative_positives = np.cumsum(ranked_targets)

        results = []

        for k in k_values:
            if isinstance(k, (bool, np.bool_)) or not isinstance(
                k, (int, np.integer)
            ):
                raise ValueError(f"K debe ser entero: {k}")

            if not 1 <= k <= n:
                raise ValueError(
                    f"K={k} fuera del rango permitido [1, {n}]."
                )

            tp = int(cumulative_positives[k - 1])
            precision = tp / k
            recall = tp / total_positives
            lift = precision / prevalence

            results.append(
                TopKResult(
                    k=int(k),
                    true_positives=tp,
                    precision_at_k=float(precision),
                    recall_at_k=float(recall),
                    lift_at_k=float(lift),
                )
            )

        return results
