from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import json

import joblib
import numpy as np

from npl_portfolio.ml.dataset_loader import SparseDatasetLoader
from npl_portfolio.ml.model_evaluator import ModelEvaluator


ROOT = Path(__file__).resolve().parents[1]

ML_DIR = ROOT / "data" / "processed" / "ml"
ARTIFACTS_DIR = ROOT / "data" / "artifacts" / "ml"

MODEL_PATH = (
    ARTIFACTS_DIR / "baseline_logistic_regression.joblib"
)

REPORT_PATH = (
    ARTIFACTS_DIR / "baseline_evaluation.json"
)


def main() -> None:
    print("=" * 70)
    print("FASE 9.5 - EVALUACION BASELINE")
    print("=" * 70)

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Modelo no encontrado: {MODEL_PATH}"
        )

    print("\n[LOAD] Modelo entrenado")
    trainer = joblib.load(MODEL_PATH)

    if not hasattr(trainer, "predict_proba"):
        raise TypeError(
            "El artefacto no expone predict_proba(). "
            "Verificar la interfaz de BaselineTrainer."
        )

    print("[LOAD] VALIDATION")

    loader = SparseDatasetLoader(ml_dir=ML_DIR)
    validation = loader.load("validation")

    print(
        f"[OK] {validation.X.shape[0]:,} filas x "
        f"{validation.X.shape[1]:,} features"
    )

    print("\n[PREDICT] Generando probabilidades")

    probabilities = np.asarray(
        trainer.predict_proba(validation.X)
    )

    # Acepta probabilidades positivas en vector 1D
    # o matriz sklearn con columnas [clase 0, clase 1].
    if probabilities.ndim == 2:
        if probabilities.shape[1] != 2:
            raise ValueError(
                "Se esperaban probabilidades de dos clases."
            )
        probabilities = probabilities[:, 1]

    evaluator = ModelEvaluator(threshold=0.50)

    result = evaluator.evaluate(
        validation.y,
        probabilities,
    )

    print("\nMETRICAS VALIDATION")
    print("-" * 70)
    print(f"ROC-AUC:   {result.roc_auc:.6f}")
    print(f"PR-AUC:    {result.pr_auc:.6f}")
    print(f"Precision: {result.precision:.6f}")
    print(f"Recall:    {result.recall:.6f}")
    print(f"F1-score:  {result.f1:.6f}")

    print("\nMATRIZ DE CONFUSION")
    print("-" * 70)
    print(f"TN: {result.tn:,}")
    print(f"FP: {result.fp:,}")
    print(f"FN: {result.fn:,}")
    print(f"TP: {result.tp:,}")

    report = {
        "phase": "9.5",
        "model": "logistic_regression_baseline",
        "dataset": "validation",
        "evaluated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "n_samples": int(validation.X.shape[0]),
        "n_features": int(validation.X.shape[1]),
        "positive_samples": int(np.sum(validation.y)),
        "metrics": asdict(result),
    }

    ARTIFACTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REPORT_PATH.open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=2)

    print(f"\n[OK] Reporte guardado: {REPORT_PATH}")
    print("[OK] FASE 9.5 FINALIZADA")


if __name__ == "__main__":
    main()
