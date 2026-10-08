"""FASE 10.1 - Evaluación de thresholds sin reentrenamiento."""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from npl_portfolio.core.paths import ML_ARTIFACTS_DIR, ML_DATA_DIR
from npl_portfolio.ml.dataset_loader import SparseDatasetLoader


MODEL_FILES = {
    "unbalanced": "unbalanced_logistic_regression.joblib",
    "balanced": "balanced_logistic_regression.joblib",
}

COMPARISON_PATH = ML_ARTIFACTS_DIR / "logistic_models_comparison.json"
OUTPUT_PATH = ML_ARTIFACTS_DIR / "advanced_threshold_evaluation.json"

THRESHOLDS = [
    0.01, 0.02, 0.03, 0.04, 0.05,
    0.10, 0.15, 0.20, 0.25, 0.30,
    0.35, 0.40, 0.45, 0.50, 0.55,
    0.60, 0.65, 0.70, 0.75, 0.80,
    0.85, 0.90, 0.95,
]


def load_probabilities(name, filename, X):
    path = ML_ARTIFACTS_DIR / filename

    if not path.is_file():
        raise FileNotFoundError(path)

    artifact = joblib.load(path)

    if not isinstance(artifact, dict):
        raise TypeError(
            f"{name}: se esperaba un diccionario con modelo y escalador"
        )

    if "scaler" not in artifact:
        raise ValueError(f"{name}: falta el escalador")

    if name == "balanced":
        estimator = artifact["trainer"]
    else:
        estimator = artifact["model"]

    X_scaled = artifact["scaler"].transform(X)
    probabilities = np.asarray(
        estimator.predict_proba(X_scaled)
    )

    if probabilities.ndim == 2 and probabilities.shape[1] == 2:
        probabilities = probabilities[:, 1]

    if probabilities.shape != (X.shape[0],):
        raise ValueError(
            f"{name}: forma inesperada {probabilities.shape}"
        )

    if not np.isfinite(probabilities).all():
        raise ValueError(f"{name}: probabilidades no finitas")

    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError(f"{name}: probabilidades fuera de rango")

    return probabilities


def evaluate_threshold(y, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(
        y, predictions, labels=[0, 1]
    ).ravel()

    return {
        "threshold": threshold,
        "precision": float(
            precision_score(y, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(y, predictions, zero_division=0)
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "predicted_positives": int(tp + fp),
        "alert_rate": float((tp + fp) / len(y)),
        "false_positive_rate": float(
            fp / (fp + tn) if fp + tn else 0.0
        ),
    }


def verify_original_results(name, results, comparison):
    original = comparison[name]["metrics"]

    current = next(
        item for item in results
        if item["threshold"] == 0.50
    )

    for metric in ("tn", "fp", "fn", "tp"):
        expected = int(original[metric])
        obtained = current[metric]

        if expected != obtained:
            raise ValueError(
                f"{name}: {metric} no coincide. "
                f"Esperado={expected}, obtenido={obtained}"
            )

    print(f"[OK] {name}: reproduce matriz original en 0.50")


def main():
    print("=" * 70)
    print("FASE 10.1 - EVALUACION AVANZADA DE THRESHOLDS")
    print("=" * 70)

    with COMPARISON_PATH.open(encoding="utf-8") as file:
        comparison = json.load(file)

    validation = SparseDatasetLoader(
        ml_dir=ML_DATA_DIR
    ).load("validation")

    y = np.asarray(validation.y).ravel()

    if not np.isin(y, [0, 1]).all():
        raise ValueError("Las etiquetas deben ser binarias")

    print(f"\nRegistros: {len(y):,}")
    print(f"Features: {validation.X.shape[1]:,}")
    print(f"Positivos: {int(np.sum(y == 1)):,}")
    print(f"Negativos: {int(np.sum(y == 0)):,}")

    report = {
        "phase": "10.1",
        "dataset": "validation",
        "n_samples": int(len(y)),
        "n_features": int(validation.X.shape[1]),
        "positive_samples": int(np.sum(y == 1)),
        "negative_samples": int(np.sum(y == 0)),
        "thresholds": THRESHOLDS,
        "models": {},
    }

    for name, filename in MODEL_FILES.items():
        print(f"\n[MODEL] {name}")

        probabilities = load_probabilities(
            name, filename, validation.X
        )

        results = [
            evaluate_threshold(y, probabilities, threshold)
            for threshold in THRESHOLDS
        ]

        verify_original_results(name, results, comparison)

        report["models"][name] = {
            "artifact": filename,
            "results": results,
        }

        print(
            f"{'Threshold':>10} "
            f"{'Precision':>11} "
            f"{'Recall':>10} "
            f"{'F1':>10} "
            f"{'Alertas':>10}"
        )

        for item in results:
            print(
                f"{item['threshold']:>10.2f} "
                f"{item['precision']:>11.4f} "
                f"{item['recall']:>10.4f} "
                f"{item['f1']:>10.4f} "
                f"{item['predicted_positives']:>10,}"
            )

    # Solo guardar si ambos modelos superaron las verificaciones.
    temporary_path = OUTPUT_PATH.with_suffix(".json.tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, allow_nan=False)

    temporary_path.replace(OUTPUT_PATH)

    print(f"\n[OK] Reporte generado: {OUTPUT_PATH}")
    print("[OK] FASE 10.1 FINALIZADA")


if __name__ == "__main__":
    main()
