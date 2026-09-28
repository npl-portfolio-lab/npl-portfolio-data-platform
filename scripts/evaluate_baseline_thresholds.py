from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from npl_portfolio.ml.dataset_loader import (
    SparseDatasetLoader,
)


ROOT = Path(__file__).resolve().parents[1]

ML_DIR = ROOT / "data" / "processed" / "ml"

MODEL_PATH = (
    ROOT
    / "data"
    / "artifacts"
    / "ml"
    / "baseline_logistic_regression.joblib"
)


def main() -> None:
    print("=" * 90)
    print("FASE 9.4 - ANALISIS DE THRESHOLDS")
    print("=" * 90)

    trainer = joblib.load(
        MODEL_PATH
    )

    loader = SparseDatasetLoader(
        ml_dir=ML_DIR
    )

    validation = loader.load(
        "validation"
    )

    probabilities = trainer.predict_proba(
        validation.X
    )

    thresholds = [
        0.50,
        0.40,
        0.30,
        0.20,
        0.15,
        0.10,
        0.05,
    ]

    print()
    print(
        f"{'THRESHOLD':>10} "
        f"{'PRECISION':>10} "
        f"{'RECALL':>10} "
        f"{'F1':>10} "
        f"{'TN':>8} "
        f"{'FP':>8} "
        f"{'FN':>8} "
        f"{'TP':>8}"
    )

    print("-" * 90)

    for threshold in thresholds:
        predictions = (
            probabilities >= threshold
        ).astype(np.int8)

        tn, fp, fn, tp = confusion_matrix(
            validation.y,
            predictions,
            labels=[0, 1],
        ).ravel()

        precision = precision_score(
            validation.y,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            validation.y,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            validation.y,
            predictions,
            zero_division=0,
        )

        print(
            f"{threshold:>10.2f} "
            f"{precision:>10.4f} "
            f"{recall:>10.4f} "
            f"{f1:>10.4f} "
            f"{tn:>8,} "
            f"{fp:>8,} "
            f"{fn:>8,} "
            f"{tp:>8,}"
        )

    print()
    print("=" * 90)
    print("ANALISIS FINALIZADO")
    print("=" * 90)


if __name__ == "__main__":
    main()
