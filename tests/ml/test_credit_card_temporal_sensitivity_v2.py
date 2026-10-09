"""Pruebas de la auditoría temporal credit_card_balance V2."""

from pathlib import Path

import pandas as pd
import pytest

from scripts.ml.audit_credit_card_temporal_sensitivity_v2 import audit


@pytest.fixture
def source_path(tmp_path: Path) -> Path:
    data = pd.DataFrame(
        {
            "SK_ID_CURR": [
                1001, 1001, 1002, 1002, 1003
            ],
            "SK_ID_PREV": [
                2001, 2001, 2002, 2002, 2003
            ],
            "MONTHS_BALANCE": [
                -2, -1, -3, -1, -1
            ],
            "SK_DPD": [
                0, 10, 0, 0, 5
            ],
            "AMT_BALANCE": [
                100.0, 200.0, 50.0, 50.0, 80.0
            ],
            "AMT_CREDIT_LIMIT_ACTUAL": [
                1000.0, 1000.0, 500.0, 500.0, 800.0
            ],
            "AMT_PAYMENT_TOTAL_CURRENT": [
                10.0, 20.0, 5.0, 5.0, 8.0
            ],
        }
    )

    path = tmp_path / "credit_card.parquet"
    data.to_parquet(path, index=False)
    return path


def test_temporal_profile(source_path):
    result = audit(source_path)
    profile = result["temporal_profile"]

    assert profile["records"] == 5
    assert profile["clients"] == 3
    assert profile["contracts"] == 3
    assert profile["recent_records"] == 3
    assert profile["recent_clients"] == 3
    assert profile["oldest_month"] == -3
    assert profile["newest_month"] == -1
    assert profile["null_months"] == 0
    assert profile["nonhistorical_months"] == 0


def test_comparable_clients(source_path):
    sensitivity = audit(source_path)["sensitivity"]

    assert sensitivity["total_clients"] == 3
    assert sensitivity["comparable_clients"] == 2
    assert sensitivity["clients_without_history"] == 1
    assert sensitivity["changed_record_count"] == 2


def test_financial_sensitivity(source_path):
    sensitivity = audit(source_path)["sensitivity"]

    assert sensitivity["changed_avg_balance"] == 1
    assert sensitivity["changed_max_balance"] == 1
    assert sensitivity["changed_avg_credit_limit"] == 0
    assert sensitivity["changed_avg_utilization"] == 1
    assert sensitivity["changed_total_payment"] == 2


def test_delinquency_sensitivity(source_path):
    sensitivity = audit(source_path)["sensitivity"]

    assert sensitivity["changed_max_dpd"] == 1
    assert sensitivity["changed_avg_dpd"] == 1
    assert sensitivity["changed_dpd_rate"] == 1


def test_detects_nonhistorical_month(source_path):
    data = pd.read_parquet(source_path)
    data.loc[0, "MONTHS_BALANCE"] = 0
    data.to_parquet(source_path, index=False)

    result = audit(source_path)

    assert result["temporal_profile"]["nonhistorical_months"] == 1


def test_missing_required_column(source_path):
    data = pd.read_parquet(source_path)
    data = data.drop(columns=["AMT_BALANCE"])
    data.to_parquet(source_path, index=False)

    with pytest.raises(ValueError, match="Columnas faltantes"):
        audit(source_path)


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        audit(tmp_path / "missing.parquet")


def test_source_is_not_modified(source_path):
    original = source_path.read_bytes()

    audit(source_path)

    assert source_path.read_bytes() == original


def test_reproducible_results(source_path):
    first = audit(source_path)
    second = audit(source_path)

    assert first == second


def test_correct_limitations(source_path):
    result = audit(source_path)

    assert result["phase"] == "10.17"
    assert result["cutoff_month"] == -2

    limitations = " ".join(result["limitations"])

    assert "solo para clientes con historial" in limitations
    assert "No modifica ni reentrena modelos" in limitations
