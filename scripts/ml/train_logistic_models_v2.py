"""Fase 10.7: entrenamiento y comparación de modelos V2."""

import json
import warnings
from time import perf_counter

import joblib
import numpy as np

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MaxAbsScaler

from npl_portfolio.core.paths import ML_DATA_DIR, ML_ARTIFACTS_DIR
from npl_portfolio.ml.dataset_loader import SparseDatasetLoader
from npl_portfolio.ml.model_evaluator import ModelEvaluator


DATA_DIR = ML_DATA_DIR / "v2"
ARTIFACT_DIR = ML_ARTIFACTS_DIR / "v2"

REPORT_PATH = ARTIFACT_DIR / "logistic_models_comparison.json"

CONFIG = {
    "solver": "liblinear",
    "max_iter": 200,
    "tol": 1e-3,
    "random_state": 42,
}

THRESHOLD = 0.50


def get_metrics(y_true, probabilities):
    result = ModelEvaluator(
        threshold=THRESHOLD
    ).evaluate(y_true, probabilities)

    return {
        "roc_auc": float(result.roc_auc),
        "pr_auc": float(result.pr_auc),
        "precision": float(result.precision),
        "recall": float(result.recall),
        "f1": float(result.f1),
        "tn": int(result.tn),
        "fp": int(result.fp),
        "fn": int(result.fn),
        "tp": int(result.tp),
    }


def train_model(
    name,
    class_weight,
    X_train,
    y_train,
    X_validation,
    y_validation,
):
    print(f"\n[TRAIN] {name.upper()}")

    model = LogisticRegression(
        **CONFIG,
        class_weight=class_weight,
    )

    start = perf_counter()

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", ConvergenceWarning)

        model.fit(X_train, y_train)

    elapsed = perf_counter() - start

    convergence_warning = any(
        issubclass(item.category, ConvergenceWarning)
        for item in captured
    )

    if convergence_warning:
        raise RuntimeError(
            f"{name}: el modelo no alcanzó convergencia."
        )

    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    if not np.isfinite(probabilities).all():
        raise ValueError(
            f"{name}: probabilidades no finitas."
        )

    metrics = get_metrics(
        y_validation,
        probabilities,
    )

    print(f"[OK] Tiempo: {elapsed:.2f} segundos")
    print(f"[OK] Iteraciones: {model.n_iter_[0]}")
    print(f"[OK] ROC-AUC: {metrics['roc_auc']:.6f}")
    print(f"[OK] PR-AUC: {metrics['pr_auc']:.6f}")

    return model, {
        "class_weight": class_weight,
        "training_seconds": elapsed,
        "iterations": int(model.n_iter_[0]),
        "convergence_warning": False,
        "metrics": metrics,
    }


def main():
    print("=" * 70)
    print("FASE 10.7 - ENTRENAMIENTO LOGISTICO V2")
    print("=" * 70)

    paths = {
        "unbalanced": ARTIFACT_DIR
        / "unbalanced_logistic_regression.joblib",
        "balanced": ARTIFACT_DIR
        / "balanced_logistic_regression.joblib",
    }

    for path in (*paths.values(), REPORT_PATH):
        if path.exists():
            raise FileExistsError(
                f"Artefacto existente; no se sobrescribirá: {path}"
            )

    loader = SparseDatasetLoader(ml_dir=DATA_DIR)

    print("[LOAD] TRAIN V2")
    train = loader.load("train")

    print("[LOAD] VALIDATION V2")
    validation = loader.load("validation")

    if train.X.shape[1] != validation.X.shape[1]:
        raise ValueError(
            "TRAIN y VALIDATION tienen distintas variables."
        )

    if len(set(train.ids) & set(validation.ids)) != 0:
        raise ValueError(
            "TRAIN y VALIDATION comparten clientes."
        )

    if not np.isin(train.y, [0, 1]).all():
        raise ValueError("Etiquetas inválidas en TRAIN.")

    if not np.isin(validation.y, [0, 1]).all():
        raise ValueError("Etiquetas inválidas en VALIDATION.")

    print("\n[SCALE] MaxAbsScaler ajustado solo con TRAIN")

    scaler = MaxAbsScaler()
    X_train = scaler.fit_transform(train.X)
    X_validation = scaler.transform(validation.X)

    if not np.isfinite(X_train.data).all():
        raise ValueError("TRAIN contiene valores no finitos.")

    if not np.isfinite(X_validation.data).all():
        raise ValueError("VALIDATION contiene valores no finitos.")

    models = {}
    results = {}

    for name, weight in (
        ("unbalanced", None),
        ("balanced", "balanced"),
    ):
        model, result = train_model(
            name=name,
            class_weight=weight,
            X_train=X_train,
            y_train=train.y,
            X_validation=X_validation,
            y_validation=validation.y,
        )

        models[name] = model
        results[name] = result

    report = {
        "phase": "10.7",
        "dataset_version": "v2",
        "training_partition": "train",
        "evaluation_partition": "validation",
        "test_evaluated": False,
        "train_rows": int(len(train.y)),
        "validation_rows": int(len(validation.y)),
        "features": int(train.X.shape[1]),
        "scaler": "MaxAbsScaler",
        "configuration": {
            **CONFIG,
            "threshold": THRESHOLD,
        },
        "models": results,
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        joblib.dump(
            {
                "model": model,
                "scaler": scaler,
                "dataset_version": "v2",
                "preprocessing_contract_path": str(
                    ARTIFACT_DIR / "preprocessing_contract.joblib"
                ),
            },
            paths[name],
        )

        print(f"[SAVE] {paths[name]}")

    REPORT_PATH.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nCOMPARACION DE METRICAS")
    print("-" * 70)
    print(
        f"{'Metrica':<15}"
        f"{'Sin balanceo':>18}"
        f"{'Balanceado':>18}"
    )

    for metric in (
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "f1",
    ):
        print(
            f"{metric:<15}"
            f"{results['unbalanced']['metrics'][metric]:>18.6f}"
            f"{results['balanced']['metrics'][metric]:>18.6f}"
        )

    print(f"\n[OK] Reporte: {REPORT_PATH}")
    print("[OK] FASE 10.7 - ENTRENAMIENTO FINALIZADO")


if __name__ == "__main__":
    main()
