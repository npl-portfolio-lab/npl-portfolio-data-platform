"""Pruebas de la auditoría temporal de Bureau V2."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/ml/audit_bureau_temporal_availability_v2.py"


def load_auditor():
    spec = importlib.util.spec_from_file_location(
        "audit_bureau_temporal_availability_v2",
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

    bureau = pd.DataFrame(
        {
            "SK_ID_CURR": [1001, 1002, 1003, 1004],
            "SK_ID_BUREAU": [2001, 2002, 2003, 2004],
            "DAYS_CREDIT_UPDATE": [10, 0, -5, 372],
            "CREDIT_ACTIVE": [
                "Active", "Closed", "Active", "Active"
            ],
            "AMT_CREDIT_SUM_DEBT": [500.0, 0.0, 100.0, 300.0],
            "CREDIT_DAY_OVERDUE": [0, 0, 5, 0],
            "AMT_CREDIT_SUM_OVERDUE": [0.0, 0.0, 20.0, 0.0],
            "DAYS_CREDIT_ENDDATE": [30, -10, -5, 400],
        }
    )

    bureau_path = tmp_path / "bureau.parquet"
    bureau.to_parquet(bureau_path, index=False)

    features_path = tmp_path / "bureau_features.parquet"
    pd.DataFrame(
        {"SK_ID_CURR": [1001, 1002, 1003, 1004]}
    ).to_parquet(features_path, index=False)

    split_paths = {}

    for name, clients in {
        "TRAIN": [1001, 1002],
        "VALIDATION": [1003],
        "TEST": [1004],
    }.items():
        path = tmp_path / f"{name.lower()}.parquet"
        pd.DataFrame(
            {"SK_ID_CURR": clients}
        ).to_parquet(path, index=False)
        split_paths[name] = path

    monkeypatch.setattr(module, "BUREAU", bureau_path)
    monkeypatch.setattr(module, "BUREAU_FEATURES", features_path)
    monkeypatch.setattr(module, "SPLITS", split_paths)

    return module, split_paths, bureau_path


def test_detects_only_future_updates(audit_environment):
    module, _, _ = audit_environment

    result = module.audit()

    assert result["records"] == 2
    assert result["clients"] == 2
    assert result["credits"] == 2
    assert result["min_update"] == 10
    assert result["max_update"] == 372


def test_affected_credit_profile(audit_environment):
    module, _, _ = audit_environment

    result = module.audit()

    assert result["active"] == 2
    assert result["closed"] == 0
    assert result["with_debt"] == 2
    assert result["with_overdue_days"] == 0
    assert result["with_overdue_amount"] == 0
    assert result["future_enddate"] == 2


def test_split_distribution(audit_environment):
    module, _, _ = audit_environment

    result = module.audit()

    assert result["clients_in_bureau_features"] == 2
    assert result["splits"] == {
        "TRAIN": 1,
        "VALIDATION": 0,
        "TEST": 1,
    }


def test_rejects_split_without_client_id(audit_environment):
    module, paths, _ = audit_environment

    pd.DataFrame(
        {"OTHER_COLUMN": [1, 2]}
    ).to_parquet(paths["TRAIN"], index=False)

    with pytest.raises(ValueError, match="SK_ID_CURR"):
        module.audit()


def test_does_not_modify_source(audit_environment):
    module, _, bureau_path = audit_environment

    original_bytes = bureau_path.read_bytes()

    module.audit()

    assert bureau_path.read_bytes() == original_bytes


def test_rejects_missing_source(audit_environment):
    module, _, bureau_path = audit_environment

    bureau_path.unlink()

    with pytest.raises(FileNotFoundError):
        module.audit()
