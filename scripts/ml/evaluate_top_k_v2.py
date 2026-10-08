"""Fase 10.8: comparación operativa Top-K sobre VALIDATION V2."""

import json

import joblib
import numpy as np

from npl_portfolio.core.paths import ML_DATA_DIR, ML_ARTIFACTS_DIR
from npl_portfolio.ml.dataset_loader import SparseDatasetLoader
from npl_portfolio.ml.top_k_evaluator import TopKEvaluator


DATA_DIR = ML_DATA_DIR / "v2"
ARTIFACT_DIR = ML_ARTIFACTS_DIR / "v2"

OUTPUT_PATH = ARTIFACT_DIR / "top_k_comparison.json"
K_VALUES = (1000, 3000, 5000)


def main():
    print("=" * 70)
    print("FASE 10.8 - COMPARACION TOP-K V2")
    print("=" * 70)

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"El reporte ya existe y no será sobrescrito: {OUTPUT_PATH}"
        )

    loader = SparseDatasetLoader(ml_dir=DATA_DIR)

    print("[LOAD] VALIDATION V2")
    validation = loader.load("validation")

    if len(np.unique(validation.ids)) != len(validation.ids):
        raise ValueError("Existen IDs duplicados en VALIDATION.")

    evaluator = TopKEvaluator()
    model_results = {}

    for name in ("unbalanced", "balanced"):
        model_path = (
            ARTIFACT_DIR / f"{name}_logistic_regression.joblib"
        )

        artifact = joblib.load(model_path)

        if artifact.get("dataset_version") != "v2":
            raise ValueError(f"{name}: versión incorrecta.")

        model = artifact["model"]
        scaler = artifact["scaler"]

        if model.n_features_in_ != validation.X.shape[1]:
            raise ValueError(f"{name}: variables incompatibles.")

        if scaler.n_features_in_ != validation.X.shape[1]:
            raise ValueError(f"{name}: escalador incompatible.")

        print(f"\n[EVALUATE] {name.upper()}")

        X_scaled = scaler.transform(validation.X)

        probabilities = model.predict_proba(X_scaled)[:, 1]

        results = evaluator.evaluate(
            validation.y,
            probabilities,
            k_values=K_VALUES,
        )

        model_results[name] = {
            str(result.k): {
                "true_positives": result.true_positives,
                "precision_at_k": result.precision_at_k,
                "recall_at_k": result.recall_at_k,
                "lift_at_k": result.lift_at_k,
            }
            for result in results
        }

        for result in results:
            print(
                f"[PASS] K={result.k:,} "
                f"TP={result.true_positives:,} "
                f"Precision={result.precision_at_k:.4f} "
                f"Recall={result.recall_at_k:.4f} "
                f"Lift={result.lift_at_k:.2f}x"
            )

    report = {
        "phase": "10.8",
        "dataset_version": "v2",
        "evaluation_partition": "validation",
        "test_evaluated": False,
        "validation_rows": int(len(validation.y)),
        "positive_cases": int(np.sum(validation.y)),
        "prevalence": float(np.mean(validation.y)),
        "features": int(validation.X.shape[1]),
        "k_values": list(K_VALUES),
        "models": model_results,
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\n[OK] Reporte: {OUTPUT_PATH}")
    print("[OK] FASE 10.8 - EVALUACION FINALIZADA")


if __name__ == "__main__":
    main()
