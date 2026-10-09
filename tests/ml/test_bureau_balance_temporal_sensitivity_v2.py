"""Pruebas de la auditoría temporal de Bureau Balance."""

import ast
from pathlib import Path

import pytest

from npl_portfolio.features.bureau_balance_features import (
    BureauBalanceFeatureBuilder,
)


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts/ml/audit_bureau_balance_temporal_sensitivity_v2.py"
)


def load_build_queries():
    """Carga solo la función pura, sin ejecutar el auditor."""
    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)

    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "build_queries"
        ),
        None,
    )

    if function is None:
        raise AssertionError("No existe build_queries")

    module = ast.Module(body=[function], type_ignores=[])
    namespace = {}
    exec(compile(module, str(SCRIPT), "exec"), namespace)

    return namespace["build_queries"]


class FakeBuilder:
    def __init__(self, query):
        self.query = query

    def _get_query(self):
        return self.query


def test_original_query_remains_unchanged():
    build_queries = load_build_queries()
    query = "SELECT * FROM mapped_history GROUP BY SK_ID_CURR"

    original, conservative = build_queries(FakeBuilder(query))

    assert original == query
    assert "FROM mapped_history WHERE MONTHS_BALANCE < 0" in conservative


def test_temporal_filter_is_applied_once():
    build_queries = load_build_queries()
    query = "SELECT * FROM mapped_history"

    _, conservative = build_queries(FakeBuilder(query))

    assert conservative.count("MONTHS_BALANCE < 0") == 1


@pytest.mark.parametrize(
    "query",
    [
        "SELECT * FROM other_table",
        "SELECT * FROM mapped_history UNION ALL "
        "SELECT * FROM mapped_history",
    ],
)
def test_invalid_query_structure_is_rejected(query):
    build_queries = load_build_queries()

    with pytest.raises(ValueError):
        build_queries(FakeBuilder(query))


def test_feature_builder_has_nineteen_features():
    """Verifica el esquema SQL sin acceder a los datos reales."""
    builder = object.__new__(BureauBalanceFeatureBuilder)
    query = builder._get_query()

    expected = [
        "BB_RECORD_COUNT",
        "BB_BUREAU_CREDIT_COUNT",
        "BB_OLDEST_MONTH",
        "BB_RECENT_MONTH",
        "BB_STATUS_0_COUNT",
        "BB_STATUS_1_COUNT",
        "BB_STATUS_2_COUNT",
        "BB_STATUS_3_COUNT",
        "BB_STATUS_4_COUNT",
        "BB_STATUS_5_COUNT",
        "BB_STATUS_C_COUNT",
        "BB_STATUS_X_COUNT",
        "BB_DPD_RECORD_COUNT",
        "BB_SEVERE_DPD_RECORD_COUNT",
        "BB_MAX_STATUS_LEVEL",
        "BB_DPD_RATE",
        "BB_SEVERE_DPD_RATE",
        "BB_CLOSED_STATUS_RATE",
        "BB_UNKNOWN_STATUS_RATE",
    ]

    assert len(expected) == 19

    for feature in expected:
        assert f"AS {feature}" in query
