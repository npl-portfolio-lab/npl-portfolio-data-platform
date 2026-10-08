from pathlib import Path
from npl_portfolio.core.paths import PROJECT_ROOT, ML_DATA_DIR, ML_ARTIFACTS_DIR

import joblib

from npl_portfolio.ml.baseline_trainer import (
    BaselineTrainer,
)
from npl_portfolio.ml.dataset_loader import (
    SparseDatasetLoader,
)


ROOT = PROJECT_ROOT

ML_DIR = ML_DATA_DIR
ARTIFACTS_DIR = ML_ARTIFACTS_DIR

MODEL_PATH = (
    ARTIFACTS_DIR
    / "baseline_logistic_regression.joblib"
)


def print_metrics(metrics) -> None:
    tn, fp, fn, tp = (
        metrics.confusion_matrix.ravel()
    )

    print()
    print("=" * 80)
    print("METRICAS VALIDATION")
    print("=" * 80)

    print(
        f"ROC-AUC:    {metrics.roc_auc:.6f}"
    )
    print(
        f"PR-AUC:     {metrics.pr_auc:.6f}"
    )
    print(
        f"Precision:  {metrics.precision:.6f}"
    )
    print(
        f"Recall:     {metrics.recall:.6f}"
    )
    print(
        f"F1-score:   {metrics.f1:.6f}"
    )

    print()
    print("MATRIZ DE CONFUSION")
    print("-" * 80)

    print(
        f"TN: {tn:,}"
    )
    print(
        f"FP: {fp:,}"
    )
    print(
        f"FN: {fn:,}"
    )
    print(
        f"TP: {tp:,}"
    )


def main() -> None:
    print("=" * 80)
    print("FASE 9.4 - BASELINE LOGISTIC REGRESSION")
    print("=" * 80)

    loader = SparseDatasetLoader(
        ml_dir=ML_DIR
    )

    print()
    print("[LOAD] TRAIN")

    train = loader.load(
        "train"
    )

    print(
        f"[OK] TRAIN: "
        f"{train.X.shape[0]:,} filas x "
        f"{train.X.shape[1]:,} features"
    )

    print(
        f"[OK] TARGET positivos TRAIN: "
        f"{train.y.sum():,} "
        f"({train.y.mean():.4%})"
    )

    print()
    print("[LOAD] VALIDATION")

    validation = loader.load(
        "validation"
    )

    print(
        f"[OK] VALIDATION: "
        f"{validation.X.shape[0]:,} filas x "
        f"{validation.X.shape[1]:,} features"
    )

    print(
        f"[OK] TARGET positivos VALIDATION: "
        f"{validation.y.sum():,} "
        f"({validation.y.mean():.4%})"
    )

    if (
        train.X.shape[1]
        != validation.X.shape[1]
    ):
        raise ValueError(
            "TRAIN y VALIDATION tienen "
            "diferente número de features."
        )

    print()
    print(
        "[TRAIN] Entrenando LogisticRegression..."
    )

    trainer = BaselineTrainer(
        random_state=42,
        threshold=0.50,
    )

    trainer.fit(
        train.X,
        train.y,
    )

    print("[OK] Modelo entrenado.")

    print()
    print(
        "[EVALUATE] Evaluando exclusivamente "
        "sobre VALIDATION..."
    )

    metrics = trainer.evaluate(
        validation.X,
        validation.y,
    )

    print_metrics(
        metrics
    )

    ARTIFACTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        trainer,
        MODEL_PATH,
    )

    print()
    print(
        f"[OK] Modelo guardado: {MODEL_PATH}"
    )

    print()
    print("=" * 80)
    print("FASE 9.4 BASELINE FINALIZADA")
    print("=" * 80)


if __name__ == "__main__":
    main()
