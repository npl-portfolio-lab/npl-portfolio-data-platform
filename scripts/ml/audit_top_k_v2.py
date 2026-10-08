"""Auditoría reproducible de Top-K V2 y coincidencia de rankings."""

import json

import joblib
import numpy as np

from npl_portfolio.core.paths import ML_DATA_DIR, ML_ARTIFACTS_DIR
from npl_portfolio.ml.dataset_loader import SparseDatasetLoader
from npl_portfolio.ml.top_k_evaluator import TopKEvaluator


ARTIFACT_DIR = ML_ARTIFACTS_DIR / "v2"
REPORT_PATH = ARTIFACT_DIR / "top_k_comparison.json"
OVERLAP_PATH = ARTIFACT_DIR / "top_k_overlap.json"


def main():
    print("=" * 70)
    print("FASE 10.8 - AUDITORIA TOP-K V2")
    print("=" * 70)

    if OVERLAP_PATH.exists():
        raise FileExistsError(
            f"No se sobrescribirá: {OVERLAP_PATH}"
        )

    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    assert report["phase"] == "10.8"
    assert report["dataset_version"] == "v2"
    assert report["evaluation_partition"] == "validation"
    assert report["test_evaluated"] is False

    validation = SparseDatasetLoader(
        ml_dir=ML_DATA_DIR / "v2"
    ).load("validation")

    assert len(validation.y) == report["validation_rows"]
    assert int(np.sum(validation.y)) == report["positive_cases"]
    assert validation.X.shape[1] == report["features"]
    assert len(np.unique(validation.ids)) == len(validation.ids)

    k_values = report["k_values"]
    assert len(k_values) == len(set(k_values))
    assert all(
        isinstance(k, int) and not isinstance(k, bool)
        and 1 <= k <= len(validation.y)
        for k in k_values
    )

    rankings = {}

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

        rankings[name] = np.argsort(
            -probabilities,
            kind="stable",
        )

        results = TopKEvaluator().evaluate(
            validation.y,
            probabilities,
            k_values=k_values,
        )

        for result in results:
            expected = report["models"][name][str(result.k)]

            assert result.true_positives == expected["true_positives"]

            for metric in (
                "precision_at_k",
                "recall_at_k",
                "lift_at_k",
            ):
                assert np.isclose(
                    getattr(result, metric),
                    expected[metric],
                    rtol=1e-9,
                    atol=1e-12,
                ), f"{name}: diferencia en {metric}"

            print(
                f"[PASS] {name.upper()} K={result.k:,}: "
                "métricas reproducidas"
            )

    overlap_results = {}

    for k in k_values:
        first = set(validation.ids[rankings["unbalanced"][:k]])
        second = set(validation.ids[rankings["balanced"][:k]])

        shared = len(first & second)
        different_each = k - shared

        overlap_results[str(k)] = {
            "shared_clients": shared,
            "different_clients_per_model": different_each,
            "overlap_percentage": 100 * shared / k,
            "jaccard_similarity": shared / (2 * k - shared),
        }

        print(
            f"[PASS] Top {k:,}: "
            f"{shared:,} clientes compartidos "
            f"({100 * shared / k:.2f} %)"
        )

    output = {
        "phase": "10.8",
        "dataset_version": "v2",
        "evaluation_partition": "validation",
        "test_evaluated": False,
        "overlap": overlap_results,
    }

    OVERLAP_PATH.write_text(
        json.dumps(output, indent=2) + "\n",
        encoding="utf-8",
    )

    print("[PASS] TEST no utilizado")
    print("[OK] AUDITORIA TOP-K V2 COMPLETADA")


if __name__ == "__main__":
    main()
