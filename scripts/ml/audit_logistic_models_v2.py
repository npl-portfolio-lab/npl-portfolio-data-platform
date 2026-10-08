"""Fase 10.7: auditoría reproducible de modelos V2."""

import json

import joblib
import numpy as np

from npl_portfolio.core.paths import ML_DATA_DIR, ML_ARTIFACTS_DIR
from npl_portfolio.ml.dataset_loader import SparseDatasetLoader
from npl_portfolio.ml.model_evaluator import ModelEvaluator


DATA_DIR = ML_DATA_DIR / "v2"
ARTIFACT_DIR = ML_ARTIFACTS_DIR / "v2"
REPORT_PATH = ARTIFACT_DIR / "logistic_models_comparison.json"

MODEL_NAMES = ("unbalanced", "balanced")
METRICS = ("roc_auc", "pr_auc", "precision", "recall", "f1")
COUNTS = ("tn", "fp", "fn", "tp")


def main():
    print("=" * 70)
    print("FASE 10.7 - AUDITORIA DE MODELOS V2")
    print("=" * 70)

    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    assert report["phase"] == "10.7"
    assert report["dataset_version"] == "v2"
    assert report["training_partition"] == "train"
    assert report["evaluation_partition"] == "validation"
    assert report["test_evaluated"] is False

    loader = SparseDatasetLoader(ml_dir=DATA_DIR)

    # No se carga TEST.
    validation = loader.load("validation")

    assert len(validation.y) == report["validation_rows"]
    assert validation.X.shape[1] == report["features"]
    assert np.isin(validation.y, [0, 1]).all()

    threshold = report["configuration"]["threshold"]

    for name in MODEL_NAMES:
        path = ARTIFACT_DIR / f"{name}_logistic_regression.joblib"

        artifact = joblib.load(path)

        assert artifact["dataset_version"] == "v2"

        model = artifact["model"]
        scaler = artifact["scaler"]

        assert model.n_features_in_ == report["features"]
        assert scaler.n_features_in_ == report["features"]

        expected_weight = (
            "balanced" if name == "balanced" else None
        )

        assert model.class_weight == expected_weight

        transformed = scaler.transform(validation.X)

        probabilities = model.predict_proba(transformed)[:, 1]

        assert np.isfinite(probabilities).all()
        assert ((probabilities >= 0) & (probabilities <= 1)).all()

        result = ModelEvaluator(threshold=threshold).evaluate(
            validation.y,
            probabilities,
        )

        expected = report["models"][name]["metrics"]

        for metric in METRICS:
            actual = float(getattr(result, metric))

            assert np.isclose(
                actual,
                expected[metric],
                rtol=1e-8,
                atol=1e-10,
            ), f"{name}: diferencia en {metric}"

        for metric in COUNTS:
            actual = int(getattr(result, metric))

            assert actual == expected[metric], (
                f"{name}: diferencia en {metric}"
            )

        assert sum(expected[key] for key in COUNTS) == len(
            validation.y
        )

        print(f"[PASS] {name.upper()}: artefacto compatible")
        print(f"[PASS] {name.upper()}: métricas reproducidas")
        print(f"[PASS] {name.upper()}: matriz de confusión consistente")

    print("[PASS] TEST no utilizado por esta auditoría")
    print("[OK] AUDITORIA DE MODELOS V2 COMPLETADA")


if __name__ == "__main__":
    main()
