from pathlib import Path

import joblib
import numpy as np

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
    print("=" * 80)
    print("FASE 9.4 - DIAGNOSTICO PROBABILIDADES BASELINE")
    print("=" * 80)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No existe el modelo: {MODEL_PATH}"
        )

    trainer = joblib.load(MODEL_PATH)

    loader = SparseDatasetLoader(
        ml_dir=ML_DIR,
    )

    validation = loader.load(
        "validation"
    )

    probabilities = trainer.predict_proba(
        validation.X
    )

    print()
    print("DISTRIBUCION GENERAL")
    print("-" * 80)

    percentiles = [
        0,
        1,
        5,
        25,
        50,
        75,
        90,
        95,
        99,
        100,
    ]

    values = np.percentile(
        probabilities,
        percentiles,
    )

    for percentile, value in zip(
        percentiles,
        values,
        strict=True,
    ):
        print(
            f"P{percentile:>3}: {value:.6f}"
        )

    positive_probabilities = probabilities[
        validation.y == 1
    ]

    negative_probabilities = probabilities[
        validation.y == 0
    ]

    print()
    print("TARGET = 1")
    print("-" * 80)
    print(
        f"Clientes: {len(positive_probabilities):,}"
    )
    print(
        f"Media:    {positive_probabilities.mean():.6f}"
    )
    print(
        f"Mediana:  {np.median(positive_probabilities):.6f}"
    )
    print(
        f"Maximo:   {positive_probabilities.max():.6f}"
    )

    print()
    print("TARGET = 0")
    print("-" * 80)
    print(
        f"Clientes: {len(negative_probabilities):,}"
    )
    print(
        f"Media:    {negative_probabilities.mean():.6f}"
    )
    print(
        f"Mediana:  {np.median(negative_probabilities):.6f}"
    )
    print(
        f"Maximo:   {negative_probabilities.max():.6f}"
    )

    print()
    print("CASOS POR ENCIMA DEL THRESHOLD")
    print("-" * 80)

    for threshold in [
        0.50,
        0.40,
        0.30,
        0.20,
        0.10,
        0.05,
    ]:
        predicted_positive = (
            probabilities >= threshold
        )

        count = int(
            predicted_positive.sum()
        )

        percentage = (
            count / len(probabilities)
        )

        print(
            f"Threshold {threshold:.2f}: "
            f"{count:>6,} clientes "
            f"({percentage:.4%})"
        )

    print()
    print("=" * 80)
    print("DIAGNOSTICO FINALIZADO")
    print("=" * 80)


if __name__ == "__main__":
    main()
