"""Fase 10.11: calibración de probabilidades sobre VALIDATION V2."""

import json
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from npl_portfolio.core.paths import ML_DATA_DIR, ML_ARTIFACTS_DIR
from npl_portfolio.ml.dataset_loader import SparseDatasetLoader
from npl_portfolio.ml.calibration_evaluator import CalibrationEvaluator
from npl_portfolio.ml.probability_calibrator import ProbabilityCalibrator


ARTIFACT_DIR = ML_ARTIFACTS_DIR / "v2"
OUTPUT_DIR = ARTIFACT_DIR / "calibration"
REPORT_PATH = OUTPUT_DIR / "calibrators_comparison.json"

RANDOM_STATE = 42


def metrics(y, probabilities, evaluator):
    result = asdict(evaluator.evaluate(y, probabilities))

    result["roc_auc"] = float(roc_auc_score(y, probabilities))
    result["pr_auc"] = float(
        average_precision_score(y, probabilities)
    )

    return result


def main():
    print("=" * 70)
    print("FASE 10.11 - RECALIBRACION V2")
    print("=" * 70)

    planned_paths = [REPORT_PATH]

    for name in ("unbalanced", "balanced"):
        for method in ("sigmoid", "isotonic"):
            planned_paths.append(
                OUTPUT_DIR / f"{name}_{method}_calibrator.joblib"
            )

    existing = [path for path in planned_paths if path.exists()]

    if existing:
        raise FileExistsError(
            "Existen artefactos de salida. "
            "No se sobrescribirá ningún archivo: "
            + ", ".join(str(path) for path in existing)
        )

    validation = SparseDatasetLoader(
        ml_dir=ML_DATA_DIR / "v2"
    ).load("validation")

    y = validation.y
    ids = validation.ids

    if len(np.unique(ids)) != len(ids):
        raise ValueError("IDs duplicados en VALIDATION")

    indices = np.arange(len(y))

    fit_idx, eval_idx = train_test_split(
        indices,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    fit_ids = set(ids[fit_idx].tolist())
    eval_ids = set(ids[eval_idx].tolist())

    if fit_ids & eval_ids:
        raise ValueError("Fuga de clientes entre FIT y EVAL")

    if len(fit_idx) + len(eval_idx) != len(y):
        raise ValueError("Partición incompleta")

    evaluator = CalibrationEvaluator(n_bins=10)

    print(f"[PASS] FIT: {len(fit_idx):,} clientes")
    print(f"[PASS] EVAL: {len(eval_idx):,} clientes")
    print("[PASS] Sin clientes compartidos")

    report = {
        "phase": "10.11",
        "dataset_version": "v2",
        "source_partition": "validation",
        "test_evaluated": False,
        "split": {
            "method": "stratified",
            "random_state": RANDOM_STATE,
            "fit_rows": int(len(fit_idx)),
            "eval_rows": int(len(eval_idx)),
            "fit_positives": int(np.sum(y[fit_idx])),
            "eval_positives": int(np.sum(y[eval_idx])),
            "overlapping_clients": 0,
        },
        "models": {},
    }

    pending_artifacts = {}

    for name in ("unbalanced", "balanced"):
        artifact = joblib.load(
            ARTIFACT_DIR / f"{name}_logistic_regression.joblib"
        )

        if artifact.get("dataset_version") != "v2":
            raise ValueError("Versión de modelo incorrecta")

        model = artifact["model"]
        scaler = artifact["scaler"]

        if model.n_features_in_ != validation.X.shape[1]:
            raise ValueError("Dimensiones del modelo incompatibles")

        if scaler.n_features_in_ != validation.X.shape[1]:
            raise ValueError("Dimensiones del escalador incompatibles")

        probabilities = model.predict_proba(
            scaler.transform(validation.X)
        )[:, 1]

        fit_probabilities = probabilities[fit_idx]
        eval_probabilities = probabilities[eval_idx]

        model_results = {
            "original": metrics(
                y[eval_idx],
                eval_probabilities,
                evaluator,
            )
        }

        for method in ("sigmoid", "isotonic"):
            calibrator = ProbabilityCalibrator(method=method)

            calibrator.fit(
                fit_probabilities,
                y[fit_idx],
            )

            calibrated = calibrator.predict_proba(
                eval_probabilities
            )

            model_results[method] = metrics(
                y[eval_idx],
                calibrated,
                evaluator,
            )

            artifact_path = (
                OUTPUT_DIR / f"{name}_{method}_calibrator.joblib"
            )

            pending_artifacts[artifact_path] = {
                "calibrator": calibrator,
                "dataset_version": "v2",
                "source_model": name,
                "method": method,
                "fit_partition": "validation_calibration_fit",
                "random_state": RANDOM_STATE,
            }

        report["models"][name] = model_results

        print(f"\n[MODEL] {name.upper()}")

        for method, result in model_results.items():
            print(
                f"[PASS] {method:8s} "
                f"Brier={result['brier_score']:.6f} "
                f"LogLoss={result['log_loss']:.6f} "
                f"ROC-AUC={result['roc_auc']:.6f} "
                f"PR-AUC={result['pr_auc']:.6f}"
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for path, artifact in pending_artifacts.items():
        if path.exists():
            raise FileExistsError(path)
        joblib.dump(artifact, path)

    if REPORT_PATH.exists():
        raise FileExistsError(REPORT_PATH)

    REPORT_PATH.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\n[OK] Reporte: {REPORT_PATH}")
    print("[OK] TEST V2 no utilizado")
    print("[OK] FASE 10.11 FINALIZADA")


if __name__ == "__main__":
    main()
