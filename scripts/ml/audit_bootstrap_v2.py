"""Auditoría de consistencia estadística de la Fase 10.9."""

import json
import math

from npl_portfolio.core.paths import ML_ARTIFACTS_DIR


ARTIFACT_DIR = ML_ARTIFACTS_DIR / "v2"


def load_json(filename):
    path = ARTIFACT_DIR / filename
    return json.loads(path.read_text(encoding="utf-8"))


def check_close(actual, expected, label):
    if not math.isclose(
        actual, expected, rel_tol=1e-8, abs_tol=1e-10
    ):
        raise AssertionError(
            f"{label}: esperado={expected}, obtenido={actual}"
        )


def main():
    print("=" * 70)
    print("FASE 10.9 - AUDITORIA ESTADISTICA V2")
    print("=" * 70)

    bootstrap = load_json("bootstrap_comparison.json")
    logistic = load_json("logistic_models_comparison.json")
    top_k = load_json("top_k_comparison.json")

    assert bootstrap["phase"] == "10.9"
    assert bootstrap["dataset_version"] == "v2"
    assert bootstrap["evaluation_partition"] == "validation"
    assert bootstrap["test_evaluated"] is False
    assert bootstrap["comparison"] == "balanced_minus_unbalanced"
    assert bootstrap["method"] == "paired_stratified_bootstrap"
    assert bootstrap["interval_method"] == "percentile"
    assert bootstrap["iterations"] == 1000
    assert bootstrap["random_state"] == 42

    assert bootstrap["validation_rows"] == logistic["validation_rows"]
    assert bootstrap["validation_rows"] == top_k["validation_rows"]
    assert bootstrap["positive_cases"] == top_k["positive_cases"]
    assert bootstrap["features"] == logistic["features"]
    assert bootstrap["features"] == top_k["features"]

    assert logistic["test_evaluated"] is False
    assert top_k["test_evaluated"] is False

    expected_metrics = {
        "roc_auc",
        "pr_auc",
        "precision_at_k_1000",
        "precision_at_k_3000",
        "precision_at_k_5000",
    }

    assert set(bootstrap["results"]) == expected_metrics

    for name, result in bootstrap["results"].items():
        assert result["metric"] == name
        assert result["bootstrap_iterations"] == bootstrap["iterations"]

        check_close(
            result["confidence_level"],
            bootstrap["confidence_level"],
            f"{name}: confidence_level",
        )

        values = (
            result["observed_difference"],
            result["mean_difference"],
            result["ci_lower"],
            result["ci_upper"],
        )
        assert all(math.isfinite(value) for value in values)
        assert result["ci_lower"] <= result["ci_upper"]

        includes_zero = (
            result["ci_lower"] <= 0 <= result["ci_upper"]
        )
        assert result["interval_includes_zero"] == includes_zero

        if name in ("roc_auc", "pr_auc"):
            balanced = logistic["models"]["balanced"]["metrics"][name]
            unbalanced = logistic["models"]["unbalanced"]["metrics"][name]
        else:
            k = name.rsplit("_", 1)[-1]
            assert int(k) in top_k["k_values"]

            balanced = top_k["models"]["balanced"][k][
                "precision_at_k"
            ]
            unbalanced = top_k["models"]["unbalanced"][k][
                "precision_at_k"
            ]

        check_close(
            result["observed_difference"],
            balanced - unbalanced,
            f"{name}: diferencia observada",
        )

        print(f"[PASS] {name}: resultado consistente")

    print("[PASS] Coincidencia con fases 10.7 y 10.8")
    print("[PASS] TEST no utilizado")
    print("[OK] AUDITORIA ESTADISTICA V2 COMPLETADA")


if __name__ == "__main__":
    main()
