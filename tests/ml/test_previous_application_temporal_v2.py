"""Pruebas de la auditoría temporal de previous_application V2."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/ml/audit_previous_application_temporal_v2.py"


def load_auditor():
    spec = importlib.util.spec_from_file_location(
        "audit_previous_application_temporal_v2",
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

    data = pd.DataFrame(
        {
            "DAYS_DECISION": [-100, -20, -1, -5, -30],
            "DAYS_FIRST_DRAWING": [
                -80, 365243, None, 0, 5
            ],
            "DAYS_FIRST_DUE": [
                -60, 365243, None, -2, 10
            ],
            "DAYS_LAST_DUE_1ST_VERSION": [
                -10, 30, 365243, 0, None
            ],
            "DAYS_LAST_DUE": [
                -5, 365243, None, -1, 2
            ],
            "DAYS_TERMINATION": [
                -1, 365243, None, -2, 4
            ],
        }
    )

    source = tmp_path / "previous_application.parquet"
    data.to_parquet(source, index=False)

    monkeypatch.setattr(module, "SOURCE", source)
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)

    return module, source


def test_total_records_and_features(audit_environment):
    module, _ = audit_environment

    result = module.audit()

    assert result["total_records"] == 5
    assert result["feature_count"] == 17
    assert len(result["feature_columns"]) == 17


def test_decisions_before_reference(audit_environment):
    module, _ = audit_environment

    result = module.audit()
    profile = result["temporal_profiles"]["DAYS_DECISION"]

    assert result["decision_strictly_before_reference"] is True
    assert profile["negatives"] == 5
    assert profile["positive_non_sentinel"] == 0
    assert profile["sentinel_365243"] == 0


def test_classifies_future_dates_and_sentinel(audit_environment):
    module, _ = audit_environment

    result = module.audit()

    drawing = result["temporal_profiles"]["DAYS_FIRST_DRAWING"]

    assert drawing["nulls"] == 1
    assert drawing["negatives"] == 1
    assert drawing["zeros"] == 1
    assert drawing["positive_non_sentinel"] == 1
    assert drawing["sentinel_365243"] == 1

    last_due = result["temporal_profiles"][
        "DAYS_LAST_DUE_1ST_VERSION"
    ]

    assert last_due["positive_non_sentinel"] == 1
    assert last_due["sentinel_365243"] == 1
    assert last_due["zeros"] == 1


def test_detects_future_decision(tmp_path, monkeypatch):
    module = load_auditor()

    data = pd.DataFrame(
        {
            column: [-1, -2]
            for column in module.TEMPORAL_COLUMNS
        }
    )

    data["DAYS_DECISION"] = [-10, 3]

    source = tmp_path / "previous_application.parquet"
    data.to_parquet(source, index=False)

    monkeypatch.setattr(module, "SOURCE", source)
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)

    result = module.audit()

    assert result["decision_strictly_before_reference"] is False
    assert result["temporal_profiles"][
        "DAYS_DECISION"
    ]["positive_non_sentinel"] == 1


def test_rejects_missing_temporal_column(
    audit_environment,
):
    module, source = audit_environment

    data = pd.read_parquet(source)
    data = data.drop(columns=["DAYS_TERMINATION"])
    data.to_parquet(source, index=False)

    with pytest.raises(ValueError, match="Columnas temporales"):
        module.audit()


def test_rejects_missing_source(audit_environment):
    module, source = audit_environment

    source.unlink()

    with pytest.raises(FileNotFoundError):
        module.audit()


def test_source_remains_unchanged(audit_environment):
    module, source = audit_environment

    original = source.read_bytes()

    module.audit()

    assert source.read_bytes() == original
