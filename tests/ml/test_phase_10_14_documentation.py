"""Pruebas de validación de splits de la Fase 10.14."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "scripts/reporting/generate_phase_10_14_documentation.py"
)


def load_generator():
    spec = importlib.util.spec_from_file_location(
        "phase_10_14_documentation",
        SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_environment(tmp_path, assignments):
    bureau_path = tmp_path / "bureau.parquet"

    pd.DataFrame(
        {
            "SK_ID_CURR": [1001, 1002],
            "DAYS_CREDIT_UPDATE": [10, 20],
        }
    ).to_parquet(bureau_path, index=False)

    splits = {}

    for name, clients in assignments.items():
        path = tmp_path / f"{name.lower()}.parquet"

        pd.DataFrame(
            {"SK_ID_CURR": clients},
            dtype="int64",
        ).to_parquet(path, index=False)

        splits[name] = path

    return SimpleNamespace(
        BUREAU=bureau_path,
        SPLITS=splits,
    )


def test_valid_split_coverage(tmp_path):
    generator = load_generator()

    environment = make_environment(
        tmp_path,
        {
            "TRAIN": [1001],
            "VALIDATION": [1002],
            "TEST": [],
        },
    )

    result = generator.validate_splits(environment, 2)

    assert result["clients_covered"] == 2
    assert result["clients_in_multiple_splits"] == 0
    assert result["exclusive_and_complete"] is True


def test_rejects_duplicate_client_across_splits(tmp_path):
    generator = load_generator()

    environment = make_environment(
        tmp_path,
        {
            "TRAIN": [1001],
            "VALIDATION": [1001, 1002],
            "TEST": [],
        },
    )

    with pytest.raises(AssertionError, match="más de un split"):
        generator.validate_splits(environment, 2)


def test_rejects_incomplete_split_coverage(tmp_path):
    generator = load_generator()

    environment = make_environment(
        tmp_path,
        {
            "TRAIN": [1001],
            "VALIDATION": [],
            "TEST": [],
        },
    )

    with pytest.raises(AssertionError, match="Cobertura incompleta"):
        generator.validate_splits(environment, 2)
