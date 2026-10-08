"""FASE 10.4 - Bootstrap pareado para evaluar estabilidad Top-K."""

import json

import numpy as np

from npl_portfolio.core.paths import ML_ARTIFACTS_DIR, ML_DATA_DIR
from npl_portfolio.ml.dataset_loader import SparseDatasetLoader

from evaluate_top_k import MODEL_FILES, load_probabilities


OUTPUT_PATH = ML_ARTIFACTS_DIR / "bootstrap_stability_evaluation.json"

CAPACITIES = [1000, 3000, 5000]
N_BOOTSTRAP = 1000
RANDOM_STATE = 42
CONFIDENCE_LEVEL = 0.95


def top_k_metrics(y, scores, k):
    """Calcula métricas Top-K usando orden descendente estable."""

    if len(y) != len(scores):
        raise ValueError("Etiquetas y probabilidades incompatibles")

    if not 1 <= k <= len(y):
        raise ValueError("K fuera de rango")

    ranking = np.argsort(-scores, kind="stable")[:k]

    tp = int(np.sum(y[ranking] == 1))
    positives = int(np.sum(y == 1))

    return {
        "tp": tp,
        "precision": tp / k,
        "recall": tp / positives if positives else 0.0,
    }


def percentile_interval(values):
    """Intervalo bootstrap percentil bilateral del 95 %."""

    alpha = 1.0 - CONFIDENCE_LEVEL

    lower, upper = np.quantile(
        values,
        [alpha / 2, 1 - alpha / 2],
    )

    return {
        "lower": float(lower),
        "upper": float(upper),
    }


def bootstrap_comparison(y, scores_unbalanced, scores_balanced):
    """Compara modelos usando las mismas muestras bootstrap."""

    n = len(y)

    if not (
        len(scores_unbalanced) == n
        and len(scores_balanced) == n
    ):
        raise ValueError("Dimensiones incompatibles")

    rng = np.random.default_rng(RANDOM_STATE)

    results = {}

    for k in CAPACITIES:
        results[k] = {
            "unbalanced_precision": [],
            "balanced_precision": [],
            "unbalanced_recall": [],
            "balanced_recall": [],
            "difference_precision": [],
            "difference_recall": [],
        }

    for iteration in range(N_BOOTSTRAP):
        indices = rng.integers(0, n, size=n)

        y_sample = y[indices]
        unbalanced_sample = scores_unbalanced[indices]
        balanced_sample = scores_balanced[indices]

        for k in CAPACITIES:
            unbalanced = top_k_metrics(
                y_sample,
                unbalanced_sample,
                k,
            )

            balanced = top_k_metrics(
                y_sample,
                balanced_sample,
                k,
            )

            result = results[k]

            result["unbalanced_precision"].append(
                unbalanced["precision"]
            )
            result["balanced_precision"].append(
                balanced["precision"]
            )

            result["unbalanced_recall"].append(
                unbalanced["recall"]
            )
            result["balanced_recall"].append(
                balanced["recall"]
            )

            result["difference_precision"].append(
                balanced["precision"] - unbalanced["precision"]
            )
            result["difference_recall"].append(
                balanced["recall"] - unbalanced["recall"]
            )

        if (iteration + 1) % 200 == 0:
            print(
                f"[BOOTSTRAP] {iteration + 1}/{N_BOOTSTRAP}"
            )

    summary = {}

    for k, metrics in results.items():
        summary[str(k)] = {}

        for name, values in metrics.items():
            array = np.asarray(values, dtype=float)

            summary[str(k)][name] = {
                "mean": float(np.mean(array)),
                "confidence_interval": percentile_interval(array),
            }

        ci = summary[str(k)]["difference_precision"][
            "confidence_interval"
        ]

        summary[str(k)]["precision_difference_excludes_zero"] = (
            ci["lower"] > 0 or ci["upper"] < 0
        )

    return summary


def main():
    print("=" * 70)
    print("FASE 10.4 - ESTABILIDAD ESTADISTICA TOP-K")
    print("=" * 70)

    validation = SparseDatasetLoader(
        ml_dir=ML_DATA_DIR
    ).load("validation")

    y = np.asarray(validation.y).ravel()

    if not np.isin(y, [0, 1]).all():
        raise ValueError("TARGET debe ser binario")

    if not 0 < np.sum(y) < len(y):
        raise ValueError("VALIDATION debe contener ambas clases")

    scores = {}

    for name, filename in MODEL_FILES.items():
        print(f"[LOAD] {name}")

        scores[name] = load_probabilities(
            name,
            filename,
            validation.X,
        )

    print(f"\nRegistros: {len(y):,}")
    print(f"Repeticiones: {N_BOOTSTRAP:,}")
    print(f"Capacidades: {CAPACITIES}")

    summary = bootstrap_comparison(
        y,
        scores["unbalanced"],
        scores["balanced"],
    )

    report = {
        "phase": "10.4",
        "dataset": "validation",
        "n_samples": int(len(y)),
        "n_bootstrap": N_BOOTSTRAP,
        "random_state": RANDOM_STATE,
        "confidence_level": CONFIDENCE_LEVEL,
        "method": "paired_nonparametric_bootstrap_percentile",
        "difference_definition": "balanced_minus_unbalanced",
        "capacities": CAPACITIES,
        "results": summary,
    }

    for k in CAPACITIES:
        result = summary[str(k)]

        diff = result["difference_precision"]
        ci = diff["confidence_interval"]

        print(f"\nTOP-{k:,}")
        print(f"Diferencia media Precision@K: {diff['mean']:.6f}")
        print(
            "IC 95 %: "
            f"[{ci['lower']:.6f}, {ci['upper']:.6f}]"
        )
        print(
            "IC excluye cero:",
            result["precision_difference_excludes_zero"],
        )

    temporary_path = OUTPUT_PATH.with_suffix(".json.tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=2,
            allow_nan=False,
        )

    temporary_path.replace(OUTPUT_PATH)

    print(f"\n[OK] Reporte generado: {OUTPUT_PATH}")
    print("[OK] FASE 10.4 FINALIZADA")


if __name__ == "__main__":
    main()
