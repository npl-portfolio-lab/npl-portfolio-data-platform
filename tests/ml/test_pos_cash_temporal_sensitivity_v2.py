"""Pruebas de la auditoría temporal POS_CASH V2."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/ml/audit_pos_cash_temporal_sensitivity_v2.py"


def load_auditor():
    spec = importlib.util.spec_from_file_location(
        "audit_pos_cash_temporal_sensitivity_v2",
        SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def audit_environment(tmp_path, monkeypatch):
    module = load_auditor()

    source = pd.DataFrame(
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
        }
    )

    path = tmp_path / "pos_cash.parquet"
    source.to_parquet(path, index=False)

    monkeypatch.setattr(module, "SOURCE", path)

    return module, path


def test_temporal_profile(audit_environment):
    module, _ = audit_environment

    result = module.audit()
    profile = result["temporal_profile"]

    assert profile["records"] == 5
    assert profile["clients"] == 3
    assert profile["contracts"] == 3
    assert profile["recent_records"] == 3
    assert profile["recent_clients"] == 3
    assert profile["oldest_month"] == -3
    assert profile["newest_month"] == -1
    assert profile["nonhistorical_months"] == 0


def test_temporal_sensitivity(audit_environment):
    module, _ = audit_environment

    result = module.audit()
    sensitivity = result["sensitivity"]

    assert sensitivity["total_clients"] == 3
    assert sensitivity["clients_without_history"] == 1
    assert sensitivity["changed_record_count"] == 3
    assert sensitivity["changed_max_dpd"] == 2
    assert sensitivity["changed_avg_dpd"] == 2
    assert sensitivity["changed_dpd_rate"] == 2


def test_detects_nonhistorical_month(tmp_path, monkeypatch):
    module = load_auditor()

    source = pd.DataFrame(
        {
            "SK_ID_CURR": [1001, 1001],
            "SK_ID_PREV": [2001, 2001],
            "MONTHS_BALANCE": [-1, 0],
            "SK_DPD": [0, 5],
        }
    )

    path = tmp_path / "pos_cash.parquet"
    source.to_parquet(path, index=False)

    monkeypatch.setattr(module, "SOURCE", path)

    result = module.audit()

    assert result["temporal_profile"]["nonhistorical_months"] == 1


def test_rejects_missing_column(audit_environment):
    module, path = audit_environment

    data = pd.read_parquet(path)
    data = data.drop(columns=["SK_DPD"])
    data.to_parquet(path, index=False)

    with pytest.raises(ValueError, match="Columnas faltantes"):
        module.audit()


def test_rejects_missing_source(audit_environment):
    module, path = audit_environment

    path.unlink()

    with pytest.raises(FileNotFoundError):
        module.audit()


def test_source_remains_unchanged(audit_environment):
    module, path = audit_environment

    original_bytes = path.read_bytes()

    module.audit()

    assert path.read_bytes() == original_bytes
