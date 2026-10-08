from dataclasses import asdict
from pathlib import Path
from time import perf_counter
import json
import warnings

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import MaxAbsScaler

from npl_portfolio.ml.balanced_trainer import BalancedTrainer
from npl_portfolio.ml.dataset_loader import SparseDatasetLoader
from npl_portfolio.ml.model_evaluator import ModelEvaluator

ROOT = Path(__file__).resolve().parents[1]

ML_DIR = ROOT / "data" / "processed" / "ml"
ARTIFACTS_DIR = ROOT / "data" / "artifacts" / "ml"

MODEL_PATH = ARTIFACTS_DIR / "balanced_logistic_regression.joblib"
REPORT_PATH = ARTIFACTS_DIR / "balanced_evaluation.json"


def main() -> None:
    print("=" * 70)
    print("FASE 9.6 - LOGISTIC REGRESSION BALANCED")
    print("=" * 70)

    loader = SparseDatasetLoader(ml_dir=ML_DIR)

    print("\n[LOAD] TRAIN")
    train = loader.load("train")

    print(f"[OK] TRAIN: {train.X.shape[0]:,} filas x " f"{train.X.shape[1]:,} features")

    print("\n[LOAD] VALIDATION")
    validation = loader.load("validation")

    print(
        f"[OK] VALIDATION: {validation.X.shape[0]:,} filas x "
        f"{validation.X.shape[1]:,} features"
    )

    if train.X.shape[1] != validation.X.shape[1]:
        raise ValueError("TRAIN y VALIDATION tienen diferente número de features.")

    # Escalado: aprender exclusivamente con TRAIN.
    print("\n[SCALE] Escalando TRAIN")

    scaler = MaxAbsScaler()
    X_train_scaled = scaler.fit_transform(train.X)

    print("[OK] TRAIN escalado")

    # Configuración del modelo.
    trainer = BalancedTrainer(
        random_state=42,
        threshold=0.50,
    )

    trainer.model.set_params(
        max_iter=200,
        tol=1e-3,
    )

    print("\n[TRAIN] Entrenando modelo balanceado")

    start = perf_counter()

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", ConvergenceWarning)
        trainer.fit(X_train_scaled, train.y)

    elapsed = perf_counter() - start

    convergence_warnings = [
        str(w.message) for w in captured if issubclass(w.category, ConvergenceWarning)
    ]

    print("[OK] Entrenamiento finalizado")
    print(f"[OK] Tiempo: {elapsed:.2f} segundos")
    print(f"[OK] Iteraciones: {trainer.model.n_iter_[0]}")

    if convergence_warnings:
        print("[WARNING] El optimizador reportó falta de convergencia")
        for message in convergence_warnings:
            print(message)

    # Evaluación: aplicar el mismo escalador a VALIDATION.
    print("\n[EVALUATE] VALIDATION")

    X_validation_scaled = scaler.transform(validation.X)

    probabilities = trainer.predict_proba(X_validation_scaled)

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
        "phase": "9.6",
        "model": "logistic_regression_balanced",
        "dataset": "validation",
        "n_samples": int(validation.X.shape[0]),
        "n_features": int(validation.X.shape[1]),
        "positive_samples": int(np.sum(validation.y)),
        "class_weight": "balanced",
        "scaler": "MaxAbsScaler",
        "solver": trainer.model.solver,
        "max_iter": trainer.model.max_iter,
        "tol": trainer.model.tol,
        "training_seconds": elapsed,
        "iterations": int(trainer.model.n_iter_[0]),
        "convergence_warning": bool(convergence_warnings),
        "metrics": asdict(result),
    }

    ARTIFACTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Guardar modelo y escalador juntos.
    joblib.dump(
        {
            "trainer": trainer,
            "scaler": scaler,
        },
        MODEL_PATH,
    )

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(f"\n[OK] Modelo: {MODEL_PATH}")
    print(f"[OK] Reporte: {REPORT_PATH}")
    print("[OK] FASE 9.6 - EXPERIMENTO FINALIZADO")


if __name__ == "__main__":
    main()
