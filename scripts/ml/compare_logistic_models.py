from pathlib import Path
from time import perf_counter
import json
import joblib
import warnings

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MaxAbsScaler

from npl_portfolio.ml.dataset_loader import SparseDatasetLoader
from npl_portfolio.ml.model_evaluator import ModelEvaluator
from npl_portfolio.core.paths import ML_DATA_DIR, ML_ARTIFACTS_DIR

ML_DIR = ML_DATA_DIR
ARTIFACTS_DIR = ML_ARTIFACTS_DIR

ML_DIR = ML_DATA_DIR
ARTIFACTS_DIR = ML_ARTIFACTS_DIR

BALANCED_PATH = ARTIFACTS_DIR / "balanced_evaluation.json"
OUTPUT_PATH = ARTIFACTS_DIR / "logistic_models_comparison.json"
UNBALANCED_MODEL_PATH = ARTIFACTS_DIR / "unbalanced_logistic_regression.joblib"


def main() -> None:
    print("=" * 70)
    print("FASE 9.6 - COMPARACION CONTROLADA DE MODELOS")
    print("=" * 70)

    if not BALANCED_PATH.exists():
        raise FileNotFoundError(BALANCED_PATH)

    with BALANCED_PATH.open(encoding="utf-8") as file:
        balanced = json.load(file)

    # Validar que el experimento previo sea comparable.
    expected = {
        "scaler": "MaxAbsScaler",
        "solver": "liblinear",
        "max_iter": 200,
        "tol": 1e-3,
        "class_weight": "balanced",
    }

    for key, value in expected.items():
        if balanced.get(key) != value:
            raise ValueError(
                f"Configuracion incompatible: {key}. "
                f"Esperado={value}, obtenido={balanced.get(key)}"
            )

    loader = SparseDatasetLoader(ml_dir=ML_DIR)

    print("\n[LOAD] TRAIN")
    train = loader.load("train")

    print("[LOAD] VALIDATION")
    validation = loader.load("validation")

    if train.X.shape[1] != validation.X.shape[1]:
        raise ValueError("TRAIN y VALIDATION tienen distintas features.")

    print("\n[SCALE] Ajustando MaxAbsScaler con TRAIN")

    scaler = MaxAbsScaler()
    X_train = scaler.fit_transform(train.X)
    X_validation = scaler.transform(validation.X)

    model = LogisticRegression(
        solver="liblinear",
        max_iter=200,
        tol=1e-3,
        random_state=42,
        class_weight=None,
    )

    print("\n[TRAIN] Regresion Logistica sin balanceo")

    start = perf_counter()

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X_train, train.y)

    elapsed = perf_counter() - start

    convergence_warning = any(
        issubclass(w.category, ConvergenceWarning) for w in captured
    )

    print(f"[OK] Tiempo: {elapsed:.2f} segundos")
    print(f"[OK] Iteraciones: {model.n_iter_[0]}")

    if convergence_warning:
        print("[WARNING] El modelo reporto falta de convergencia.")

    print("\n[EVALUATE] VALIDATION")

    probabilities = model.predict_proba(X_validation)[:, 1]

    result = ModelEvaluator(threshold=0.50).evaluate(
        validation.y,
        probabilities,
    )

    metrics = {
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

    print("\nCOMPARACION DE METRICAS")
    print("-" * 70)
    print(f"{'Metrica':<15} {'Sin balanceo':>15} {'Balanceado':>15}")

    for key in ("roc_auc", "pr_auc", "precision", "recall", "f1"):
        print(
            f"{key:<15} " f"{metrics[key]:>15.6f} " f"{balanced['metrics'][key]:>15.6f}"
        )

    print("\nMATRIZ DE CONFUSION")
    print("-" * 70)

    for key in ("tn", "fp", "fn", "tp"):
        print(
            f"{key.upper():<15} "
            f"{metrics[key]:>15,} "
            f"{balanced['metrics'][key]:>15,}"
        )

    report = {
        "phase": "9.6",
        "comparison": "class_weight balanced vs None",
        "configuration": {
            "scaler": "MaxAbsScaler",
            "solver": "liblinear",
            "max_iter": 200,
            "tol": 1e-3,
            "random_state": 42,
            "threshold": 0.50,
        },
        "unbalanced": {
            "training_seconds": elapsed,
            "iterations": int(model.n_iter_[0]),
            "convergence_warning": convergence_warning,
            "metrics": metrics,
        },
        "balanced": {
            "training_seconds": balanced.get("training_seconds"),
            "iterations": balanced.get("iterations"),
            "convergence_warning": balanced.get("convergence_warning"),
            "metrics": balanced["metrics"],
        },
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Guardar el modelo y el escalador utilizados en la Fase 9.6.
    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
        },
        UNBALANCED_MODEL_PATH,
    )

    print(f"[OK] Modelo guardado: {UNBALANCED_MODEL_PATH}")

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(f"\n[OK] Comparacion guardada: {OUTPUT_PATH}")
    print("[OK] FASE 9.6 - COMPARACION FINALIZADA")


if __name__ == "__main__":
    main()
