"""Fase 10.10: evaluación de calibración original V2."""

import json
from dataclasses import asdict

import joblib
import numpy as np

from npl_portfolio.core.paths import ML_DATA_DIR, ML_ARTIFACTS_DIR
from npl_portfolio.ml.dataset_loader import SparseDatasetLoader
from npl_portfolio.ml.calibration_evaluator import CalibrationEvaluator


ARTIFACT_DIR = ML_ARTIFACTS_DIR / "v2"
OUTPUT_PATH = ARTIFACT_DIR / "calibration_evaluation.json"


def main():
    print("=" * 70)
    print("FASE 10.10 - EVALUACION DE CALIBRACION V2")
    print("=" * 70)

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"El reporte ya existe: {OUTPUT_PATH}"
        )

    validation = SparseDatasetLoader(
        ml_dir=ML_DATA_DIR / "v2"
    ).load("validation")

    if len(np.unique(validation.ids)) != len(validation.ids):
        raise ValueError("IDs duplicados en VALIDATION")

    evaluator = CalibrationEvaluator(n_bins=10)

    results = {}

    for name in ("unbalanced", "balanced"):
        artifact = joblib.load(
            ARTIFACT_DIR / f"{name}_logistic_regression.joblib"
        )

        if artifact.get("dataset_version") != "v2":
            raise ValueError(f"{name}: versión incorrecta")

        model = artifact["model"]
        scaler = artifact["scaler"]

        if model.n_features_in_ != validation.X.shape[1]:
            raise ValueError(f"{name}: variables incompatibles")

        if scaler.n_features_in_ != validation.X.shape[1]:
            raise ValueError(f"{name}: escalador incompatible")

        probabilities = model.predict_proba(
            scaler.transform(validation.X)
        )[:, 1]

        result = evaluator.evaluate(
            validation.y,
            probabilities,
        )

        results[name] = asdict(result)

        print(f"\n[MODEL] {name.upper()}")
        print(f"[PASS] Brier Score: {result.brier_score:.6f}")
        print(f"[PASS] Log Loss: {result.log_loss:.6f}")
        print(
            "[PASS] Probabilidad media estimada: "
            f"{result.mean_predicted_probability:.4%}"
        )
        print(
            "[PASS] Prevalencia observada: "
            f"{result.observed_prevalence:.4%}"
        )

    report = {
        "phase": "10.10",
        "dataset_version": "v2",
        "evaluation_partition": "validation",
        "test_evaluated": False,
        "calibration_applied": False,
        "n_bins": 10,
        "validation_rows": int(len(validation.y)),
        "positive_cases": int(np.sum(validation.y)),
        "features": int(validation.X.shape[1]),
        "models": results,
    }

    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\n[OK] Reporte: {OUTPUT_PATH}")
    print("[OK] FASE 10.10 - EVALUACION FINALIZADA")


if __name__ == "__main__":
    main()
