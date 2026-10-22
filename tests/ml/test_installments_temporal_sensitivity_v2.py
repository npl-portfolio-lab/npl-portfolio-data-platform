"""Pruebas de la auditoría temporal de installments V2."""

from pathlib import Path

import duckdb
import pytest

from scripts.ml.audit_installments_temporal_sensitivity_v2 import (
    CUTOFF_DAY,
    REQUIRED_COLUMNS,
    audit,
)


def make_parquet(tmp_path: Path, rows: list[tuple]) -> Path:
    path = tmp_path / "installments.parquet"
    con = duckdb.connect()
    try:
        con.execute("""
            CREATE TABLE installments (
                SK_ID_CURR BIGINT,
                SK_ID_PREV BIGINT,
                NUM_INSTALMENT_VERSION DOUBLE,
                NUM_INSTALMENT_NUMBER BIGINT,
                DAYS_INSTALMENT DOUBLE,
                DAYS_ENTRY_PAYMENT DOUBLE,
                AMT_INSTALMENT DOUBLE,
                AMT_PAYMENT DOUBLE
            )
        """)
        con.executemany(
            "INSERT INTO installments VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        con.execute(
            "COPY installments TO ? (FORMAT PARQUET)",
            [str(path)],
        )
    finally:
        con.close()
    return path


def test_cutoff():
    assert CUTOFF_DAY == -2


def test_required_columns():
    assert len(REQUIRED_COLUMNS) == 8


def test_missing_source(tmp_path):
    with pytest.raises(FileNotFoundError):
        audit(tmp_path / "missing.parquet")


def test_no_recent_records(tmp_path):
    path = make_parquet(tmp_path, [
        (1, 10, 1.0, 1, -10.0, -9.0, 100.0, 100.0),
        (1, 10, 1.0, 2, -8.0, -7.0, 100.0, 90.0),
    ])
    result = audit(path)

    assert result["temporal_profile"]["excluded_records"] == 0
    assert result["sensitivity"]["changed_record_count"] == 0
    assert result["sensitivity"]["changed_feature_count"] == 0


def test_recent_record_changes_features(tmp_path):
    path = make_parquet(tmp_path, [
        (1, 10, 1.0, 1, -10.0, -9.0, 100.0, 100.0),
        (1, 10, 1.0, 2, -1.0, -1.0, 100.0, 100.0),
    ])
    result = audit(path)

    assert result["temporal_profile"]["excluded_records"] == 1
    assert result["sensitivity"]["changed_record_count"] == 1
    assert result["sensitivity"]["evaluated_features"] == 30
    assert result["sensitivity"]["changed_feature_count"] > 0


def test_null_payment_date_retained(tmp_path):
    path = make_parquet(tmp_path, [
        (1, 10, 1.0, 1, -10.0, None, 100.0, None),
    ])
    result = audit(path)

    assert result["temporal_profile"]["null_payment_dates"] == 1
    assert result["temporal_profile"]["excluded_records"] == 0


def test_client_without_restricted_history(tmp_path):
    path = make_parquet(tmp_path, [
        (1, 10, 1.0, 1, -1.0, -1.0, 100.0, 100.0),
        (2, 20, 1.0, 1, -10.0, -9.0, 100.0, 100.0),
    ])
    result = audit(path)

    assert result["sensitivity"]["total_clients"] == 2
    assert result["sensitivity"]["comparable_clients"] == 1
    assert result["sensitivity"]["clients_without_history"] == 1


def test_relative_source_path(tmp_path):
    path = make_parquet(tmp_path, [
        (1, 10, 1.0, 1, -10.0, -9.0, 100.0, 100.0),
    ])
    result = audit(path)

    assert result["source"] == (
        "data/processed/home_credit/"
        "installments_payments.parquet"
    )


def test_no_target_dependency():
    assert "TARGET" not in REQUIRED_COLUMNS
