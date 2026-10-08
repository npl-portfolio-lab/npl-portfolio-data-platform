"""Auditoría de independencia y reproducibilidad de calibradores V2."""

import json
import math

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from npl_portfolio.core.paths import ML_DATA_DIR, ML_ARTIFACTS_DIR
from npl_portfolio.ml.dataset_loader import SparseDatasetLoader
from npl_portfolio.ml.calibration_evaluator import CalibrationEvaluator
from npl_portfolio.ml.probability_calibrator import ProbabilityCalibrator


BASE = ML_ARTIFACTS_DIR / "v2"
CALIBRATION_DIR = BASE / "calibration"
REPORT_PATH = CALIBRATION_DIR / "calibrators_comparison.json"


def assert_close(actual, expected, label):
    if not math.isclose(
        float(actual),
        float(expected),
        rel_tol=1e-8,
        abs_tol=1e-10,
    ):
        raise AssertionError(
            f"{label}: esperado={expected}, obtenido={actual}"
        )


def verify_metrics(y, probabilities, expected, label):
    evaluator = CalibrationEvaluator(n_bins=10)
    result = evaluator.evaluate(y, probabilities)

    calculated = {
        "brier_score": result.brier_score,
        "log_loss": result.log_loss,
        "roc_auc": roc_auc_score(y, probabilities),
        "pr_auc": average_precision_score(y, probabilities),
        "mean_predicted_probability": result.mean_predicted_probability,
        "observed_prevalence": result.observed_prevalence,
    }

    for metric, value in calculated.items():
        assert_close(value, expected[metric], f"{label}: {metric}")

    assert sum(item["count"] for item in result.bins) == len(y)
    assert len(result.bins) == len(expected["bins"])

    for actual_bin, expected_bin in zip(
        result.bins, expected["bins"], strict=True
    ):
        assert actual_bin["bin_index"] == expected_bin["bin_index"]
        assert actual_bin["count"] == expected_bin["count"]
        assert actual_bin["positive_cases"] == expected_bin["positive_cases"]

        for key in (
            "lower_bound",
            "upper_bound",
            "mean_predicted_probability",
            "observed_positive_rate",
        ):
            assert_close(
                actual_bin[key],
                expected_bin[key],
                f"{label}: bin {actual_bin['bin_index']} {key}",
            )


def main():
    print("=" * 70)
    print("FASE 10.11 - AUDITORIA DE CALIBRADORES V2")
    print("=" * 70)

    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    assert report["phase"] == "10.11"
    assert report["dataset_version"] == "v2"
    assert report["source_partition"] == "validation"
    assert report["test_evaluated"] is False
    assert report["split"]["method"] == "stratified"
    assert report["split"]["random_state"] == 42
    assert set(report["models"]) == {"unbalanced", "balanced"}

    dataset = SparseDatasetLoader(
        ml_dir=ML_DATA_DIR / "v2"
    ).load("validation")

    assert len(np.unique(dataset.ids)) == len(dataset.ids)

    indices = np.arange(len(dataset.y))

    fit_idx, eval_idx = train_test_split(
        indices,
        test_size=0.5,
        random_state=42,
        stratify=dataset.y,
    )

    assert len(fit_idx) == report["split"]["fit_rows"]
    assert len(eval_idx) == report["split"]["eval_rows"]
    assert int(np.sum(dataset.y[fit_idx])) == report["split"]["fit_positives"]
    assert int(np.sum(dataset.y[eval_idx])) == report["split"]["eval_positives"]

    assert len(np.intersect1d(fit_idx, eval_idx)) == 0
    assert len(np.union1d(fit_idx, eval_idx)) == len(dataset.y)

    assert len(
        np.intersect1d(dataset.ids[fit_idx], dataset.ids[eval_idx])
    ) == 0

    assert report["split"]["overlapping_clients"] == 0

    print("[PASS] Separación FIT/EVAL reproducible")
    print("[PASS] Sin clientes compartidos")

    for name in ("unbalanced", "balanced"):
        original = joblib.load(
            BASE / f"{name}_logistic_regression.joblib"
        )

        assert original["dataset_version"] == "v2"

        model = original["model"]
        scaler = original["scaler"]

        assert model.n_features_in_ == dataset.X.shape[1]
        assert scaler.n_features_in_ == dataset.X.shape[1]

        probabilities = model.predict_proba(
            scaler.transform(dataset.X)
        )[:, 1]

        verify_metrics(
            dataset.y[eval_idx],
            probabilities[eval_idx],
            report["models"][name]["original"],
            f"{name}/original",
        )

        print(f"[PASS] {name}/original")

        for method in ("sigmoid", "isotonic"):
            saved = joblib.load(
                CALIBRATION_DIR / f"{name}_{method}_calibrator.joblib"
            )

            assert saved["dataset_version"] == "v2"
            assert saved["source_model"] == name
            assert saved["method"] == method
            assert saved["fit_partition"] == "validation_calibration_fit"
            assert saved["random_state"] == 42

            calibrated = saved["calibrator"].predict_proba(
                probabilities[eval_idx]
            )

            verify_metrics(
                dataset.y[eval_idx],
                calibrated,
                report["models"][name][method],
                f"{name}/{method}",
            )

            # Reajuste independiente para verificar reproducibilidad.
            reproduced = ProbabilityCalibrator(method=method)
            reproduced.fit(
                probabilities[fit_idx],
                dataset.y[fit_idx],
            )

            np.testing.assert_allclose(
                calibrated,
                reproduced.predict_proba(probabilities[eval_idx]),
                rtol=1e-8,
                atol=1e-10,
            )

            print(f"[PASS] {name}/{method}: métricas y reajuste reproducidos")

    print("[PASS] TEST V2 no utilizado")
    print("[OK] AUDITORIA FASE 10.11 COMPLETADA")


if __name__ == "__main__":
    main()
