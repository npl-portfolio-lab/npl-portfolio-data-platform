"""Auditoría de calibración V2 sin utilizar TEST."""

import json
import math

import joblib
import numpy as np

from npl_portfolio.core.paths import ML_DATA_DIR, ML_ARTIFACTS_DIR
from npl_portfolio.ml.dataset_loader import SparseDatasetLoader
from npl_portfolio.ml.calibration_evaluator import CalibrationEvaluator


ARTIFACT_DIR = ML_ARTIFACTS_DIR / "v2"
REPORT_PATH = ARTIFACT_DIR / "calibration_evaluation.json"


def close(actual, expected, label):
    if not math.isclose(
        actual, expected, rel_tol=1e-8, abs_tol=1e-10
    ):
        raise AssertionError(f"{label}: resultado inconsistente")


def main():
    print("=" * 70)
    print("FASE 10.10 - AUDITORIA DE CALIBRACION V2")
    print("=" * 70)

    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    assert report["phase"] == "10.10"
    assert report["dataset_version"] == "v2"
    assert report["evaluation_partition"] == "validation"
    assert report["test_evaluated"] is False
    assert report["calibration_applied"] is False
    assert set(report["models"]) == {"unbalanced", "balanced"}

    validation = SparseDatasetLoader(
        ml_dir=ML_DATA_DIR / "v2"
    ).load("validation")

    assert len(validation.y) == report["validation_rows"]
    assert int(np.sum(validation.y)) == report["positive_cases"]
    assert validation.X.shape[1] == report["features"]
    assert len(np.unique(validation.ids)) == len(validation.ids)

    evaluator = CalibrationEvaluator(n_bins=report["n_bins"])

    for name in ("unbalanced", "balanced"):
        artifact = joblib.load(
            ARTIFACT_DIR / f"{name}_logistic_regression.joblib"
        )

        assert artifact["dataset_version"] == "v2"

        model = artifact["model"]
        scaler = artifact["scaler"]

        assert model.n_features_in_ == report["features"]
        assert scaler.n_features_in_ == report["features"]

        probabilities = model.predict_proba(
            scaler.transform(validation.X)
        )[:, 1]

        actual = evaluator.evaluate(
            validation.y, probabilities
        )

        expected = report["models"][name]

        for metric in (
            "brier_score",
            "log_loss",
            "mean_predicted_probability",
            "observed_prevalence",
        ):
            close(
                getattr(actual, metric),
                expected[metric],
                f"{name}: {metric}",
            )

        assert len(actual.bins) == len(expected["bins"])

        for actual_bin, expected_bin in zip(
            actual.bins, expected["bins"], strict=True
        ):
            for key in (
                "bin_index",
                "count",
                "positive_cases",
            ):
                assert actual_bin[key] == expected_bin[key]

            for key in (
                "lower_bound",
                "upper_bound",
                "mean_predicted_probability",
                "observed_positive_rate",
            ):
                close(
                    actual_bin[key],
                    expected_bin[key],
                    f"{name}: bin {actual_bin['bin_index']} {key}",
                )

        assert sum(item["count"] for item in actual.bins) == len(
            validation.y
        )

        assert sum(
            item["positive_cases"] for item in actual.bins
        ) == int(np.sum(validation.y))

        print(f"[PASS] {name.upper()}: métricas reproducidas")
        print(f"[PASS] {name.upper()}: intervalos consistentes")

    print("[PASS] TEST no utilizado")
    print("[OK] AUDITORIA DE CALIBRACION V2 COMPLETADA")


if __name__ == "__main__":
    main()
