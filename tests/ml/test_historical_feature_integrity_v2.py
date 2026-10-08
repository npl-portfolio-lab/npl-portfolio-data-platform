from pathlib import Path

import duckdb
import pytest

from scripts.ml.audit_historical_feature_integrity_v2 import (
    audit_source,
    check_unique_ids,
    quote_identifier,
)


def create_parquet(connection, path: Path, sql: str) -> None:
    connection.execute(
        f"COPY ({sql}) TO ? (FORMAT PARQUET)",
        [str(path)],
    )


@pytest.fixture
def connection():
    con = duckdb.connect()
    yield con
    con.close()


def test_quote_identifier_escapes_quotes():
    assert quote_identifier('my"column') == '"my""column"'


def test_unique_ids_pass(connection, tmp_path):
    path = tmp_path / "unique.parquet"

    create_parquet(
        connection,
        path,
        "SELECT 1 AS SK_ID_CURR UNION ALL SELECT 2",
    )

    assert check_unique_ids(connection, path) == (2, 2)


def test_duplicate_ids_fail(connection, tmp_path):
    path = tmp_path / "duplicates.parquet"

    create_parquet(
        connection,
        path,
        "SELECT 1 AS SK_ID_CURR UNION ALL SELECT 1",
    )

    with pytest.raises(AssertionError, match="Claves invalidas"):
        check_unique_ids(connection, path)


def test_null_ids_fail(connection, tmp_path):
    path = tmp_path / "nulls.parquet"

    create_parquet(
        connection,
        path,
        """
        SELECT CAST(NULL AS INTEGER) AS SK_ID_CURR
        UNION ALL SELECT 2
        """,
    )

    with pytest.raises(AssertionError, match="Claves invalidas"):
        check_unique_ids(connection, path)


@pytest.mark.parametrize(
    ("feature_value", "flag_value", "expected_values", "expected_flags"),
    [
        (10, 1, 0, 0),
        (99, 1, 1, 0),
        (10, 0, 0, 1),
    ],
)
def test_audit_detects_inconsistencies(
    connection,
    tmp_path,
    feature_value,
    flag_value,
    expected_values,
    expected_flags,
):
    source = tmp_path / "source.parquet"
    final = tmp_path / "final.parquet"

    create_parquet(
        connection,
        source,
        "SELECT 1 AS SK_ID_CURR, 10 AS HIST_FEATURE",
    )

    create_parquet(
        connection,
        final,
        f"""
        SELECT
            1 AS SK_ID_CURR,
            {feature_value} AS HIST_FEATURE,
            {flag_value} AS HAS_HISTORY

        UNION ALL

        SELECT
            2 AS SK_ID_CURR,
            NULL AS HIST_FEATURE,
            0 AS HAS_HISTORY
        """,
    )

    result = audit_source(
        connection,
        final,
        source,
        "synthetic",
        "HAS_HISTORY",
    )

    assert result.feature_count == 1
    assert result.matched_clients == 1
    assert result.missing_history_clients == 1
    assert result.value_mismatches == expected_values
    assert result.flag_mismatches == expected_flags
