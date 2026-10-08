"""FASE 10.3 - Evaluación Top-K sin reentrenamiento."""

import json

import joblib
import numpy as np

from npl_portfolio.core.paths import ML_ARTIFACTS_DIR, ML_DATA_DIR
from npl_portfolio.ml.dataset_loader import SparseDatasetLoader


MODEL_FILES = {
    "unbalanced": "unbalanced_logistic_regression.joblib",
    "balanced": "balanced_logistic_regression.joblib",
}

CAPACITIES = [1000, 3000, 5000]

OUTPUT_PATH = ML_ARTIFACTS_DIR / "top_k_evaluation.json"


def load_probabilities(name, filename, X):
    path = ML_ARTIFACTS_DIR / filename

    if not path.is_file():
        raise FileNotFoundError(path)

    artifact = joblib.load(path)

    if not isinstance(artifact, dict):
        raise TypeError(f"{name}: estructura de artefacto inválida")

    scaler = artifact["scaler"]

    estimator = (
        artifact["trainer"]
        if name == "balanced"
        else artifact["model"]
    )

    probabilities = np.asarray(
        estimator.predict_proba(scaler.transform(X))
    )

    if probabilities.ndim == 2 and probabilities.shape[1] == 2:
        probabilities = probabilities[:, 1]

    if probabilities.shape != (X.shape[0],):
        raise ValueError(f"{name}: dimensiones inválidas")

    if not np.isfinite(probabilities).all():
        raise ValueError(f"{name}: probabilidades no finitas")

    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError(f"{name}: probabilidades fuera de rango")

    return probabilities


def evaluate_top_k(y, probabilities, k):
    n = len(y)

    if not 1 <= k <= n:
        raise ValueError(f"K debe estar entre 1 y {n}")

    positives = int(np.sum(y == 1))

    if positives == 0:
        raise ValueError("No hay positivos reales para evaluar")

    # Orden descendente y desempate estable por índice original.
    ranking = np.argsort(-probabilities, kind="stable")

    selected = ranking[:k]

    tp = int(np.sum(y[selected] == 1))
    fp = int(k - tp)
    fn = int(positives - tp)
    tn = int(n - positives - fp)

    precision = tp / k
    recall = tp / positives
    prevalence = positives / n
    lift = precision / prevalence

    return {
        "k": int(k),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision_at_k": float(precision),
        "recall_at_k": float(recall),
        "lift_at_k": float(lift),
        "selected_rate": float(k / n),
        "minimum_selected_probability": float(
            probabilities[selected[-1]]
        ),
    }


def main():
    print("=" * 70)
    print("FASE 10.3 - EVALUACION TOP-K")
    print("=" * 70)

    validation = SparseDatasetLoader(
        ml_dir=ML_DATA_DIR
    ).load("validation")

    y = np.asarray(validation.y).ravel()

    if not np.isin(y, [0, 1]).all():
        raise ValueError("TARGET debe ser binario")

    n = len(y)
    positives = int(np.sum(y == 1))

    if positives == 0 or positives == n:
        raise ValueError("Se requieren ambas clases en VALIDATION")

    report = {
        "phase": "10.3",
        "dataset": "validation",
        "n_samples": int(n),
        "n_features": int(validation.X.shape[1]),
        "positive_samples": positives,
        "negative_samples": int(n - positives),
        "prevalence": float(positives / n),
        "capacities": CAPACITIES,
        "models": {},
    }

    print(f"\nRegistros: {n:,}")
    print(f"Positivos: {positives:,}")
    print(f"Prevalencia: {positives / n:.2%}")

    for name, filename in MODEL_FILES.items():
        print(f"\n[MODEL] {name}")

        probabilities = load_probabilities(
            name, filename, validation.X
        )

        results = [
            evaluate_top_k(y, probabilities, k)
            for k in CAPACITIES
        ]

        report["models"][name] = {
            "artifact": filename,
            "results": results,
        }

        print(
            f"{'K':>8} "
            f"{'TP':>8} "
            f"{'FP':>8} "
            f"{'Precision@K':>14} "
            f"{'Recall@K':>12} "
            f"{'Lift@K':>10}"
        )

        for result in results:
            print(
                f"{result['k']:>8,} "
                f"{result['tp']:>8,} "
                f"{result['fp']:>8,} "
                f"{result['precision_at_k']:>14.4f} "
                f"{result['recall_at_k']:>12.4f} "
                f"{result['lift_at_k']:>10.4f}"
            )

    temporary_path = OUTPUT_PATH.with_suffix(".json.tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, allow_nan=False)

    temporary_path.replace(OUTPUT_PATH)

    print(f"\n[OK] Reporte generado: {OUTPUT_PATH}")
    print("[OK] FASE 10.3 FINALIZADA")


if __name__ == "__main__":
    main()
