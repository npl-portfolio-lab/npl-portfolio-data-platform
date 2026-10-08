"""Fase 10.9: comparación estadística de modelos V2."""

import json
from dataclasses import asdict

import joblib
import numpy as np

from npl_portfolio.core.paths import ML_DATA_DIR, ML_ARTIFACTS_DIR
from npl_portfolio.ml.dataset_loader import SparseDatasetLoader
from npl_portfolio.ml.paired_bootstrap import PairedBootstrap


ARTIFACT_DIR = ML_ARTIFACTS_DIR / "v2"
OUTPUT_PATH = ARTIFACT_DIR / "bootstrap_comparison.json"

ITERATIONS = 1000
CONFIDENCE_LEVEL = 0.95
RANDOM_STATE = 42


def main():
    print("=" * 70)
    print("FASE 10.9 - BOOTSTRAP PAREADO V2")
    print("=" * 70)

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"El archivo ya existe: {OUTPUT_PATH}"
        )

    validation = SparseDatasetLoader(
        ml_dir=ML_DATA_DIR / "v2"
    ).load("validation")

    if len(np.unique(validation.ids)) != len(validation.ids):
        raise ValueError("IDs duplicados en VALIDATION")

    probabilities = {}

    for name in ("unbalanced", "balanced"):
        artifact = joblib.load(
            ARTIFACT_DIR / f"{name}_logistic_regression.joblib"
        )

        if artifact.get("dataset_version") != "v2":
            raise ValueError("Versión incorrecta")

        model = artifact["model"]
        scaler = artifact["scaler"]

        if model.n_features_in_ != validation.X.shape[1]:
            raise ValueError("Dimensiones incompatibles")

        if scaler.n_features_in_ != validation.X.shape[1]:
            raise ValueError("Escalador incompatible")

        probabilities[name] = model.predict_proba(
            scaler.transform(validation.X)
        )[:, 1]

    bootstrap = PairedBootstrap(
        iterations=ITERATIONS,
        confidence_level=CONFIDENCE_LEVEL,
        random_state=RANDOM_STATE,
    )

    comparisons = [
        ("roc_auc", None),
        ("pr_auc", None),
        ("precision_at_k", 1000),
        ("precision_at_k", 3000),
        ("precision_at_k", 5000),
    ]

    results = {}

    for metric, k in comparisons:
        label = metric if k is None else f"{metric}_{k}"

        print(f"\n[BOOTSTRAP] {label}", flush=True)

        result = bootstrap.compare(
            y_true=validation.y,
            probabilities_a=probabilities["unbalanced"],
            probabilities_b=probabilities["balanced"],
            metric=metric,
            k=k,
        )

        results[label] = asdict(result)

        print(
            f"[PASS] Delta={result.observed_difference:+.6f} "
            f"IC95%=[{result.ci_lower:+.6f}, "
            f"{result.ci_upper:+.6f}] "
            f"Incluye cero={result.interval_includes_zero}",
            flush=True,
        )

    report = {
        "phase": "10.9",
        "dataset_version": "v2",
        "evaluation_partition": "validation",
        "test_evaluated": False,
        "comparison": "balanced_minus_unbalanced",
        "method": "paired_stratified_bootstrap",
        "interval_method": "percentile",
        "iterations": ITERATIONS,
        "confidence_level": CONFIDENCE_LEVEL,
        "random_state": RANDOM_STATE,
        "validation_rows": int(len(validation.y)),
        "positive_cases": int(np.sum(validation.y)),
        "features": int(validation.X.shape[1]),
        "results": results,
    }

    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\n[OK] Reporte: {OUTPUT_PATH}")
    print("[OK] FASE 10.9 - BOOTSTRAP FINALIZADO")


if __name__ == "__main__":
    main()
