"""Pruebas del generador documental de la Fase 10.15."""

import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]

SCRIPT = (
    ROOT
    / "scripts/reporting/generate_phase_10_15_documentation.py"
)


def load_generator():
    spec = importlib.util.spec_from_file_location(
        "phase_10_15_documentation",
        SCRIPT,
    )

    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


@pytest.fixture
def evidence():
    return {
        "total_records": 3,
        "feature_count": 2,
        "feature_columns": [
            "PREV_APPLICATION_COUNT",
            "PREV_AVG_DAYS_DECISION",
        ],
        "decision_strictly_before_reference": True,
        "temporal_profiles": {
            "DAYS_DECISION": {
                "nulls": 0,
                "negatives": 3,
                "zeros": 0,
                "positive_non_sentinel": 0,
                "sentinel_365243": 0,
            },
            "DAYS_FIRST_DRAWING": {
                "nulls": 1,
                "negatives": 1,
                "zeros": 0,
                "positive_non_sentinel": 0,
                "sentinel_365243": 1,
            },
        },
    }


def test_accepts_valid_evidence(evidence):
    generator = load_generator()

    assert generator.validate_evidence(evidence) is True


def test_rejects_inconsistent_temporal_counts(evidence):
    generator = load_generator()
    invalid = deepcopy(evidence)

    invalid["temporal_profiles"][
        "DAYS_FIRST_DRAWING"
    ]["sentinel_365243"] = 2

    with pytest.raises(AssertionError, match="Conteo inconsistente"):
        generator.validate_evidence(invalid)


def test_rejects_duplicate_features(evidence):
    generator = load_generator()
    invalid = deepcopy(evidence)

    invalid["feature_columns"] = [
        "PREV_APPLICATION_COUNT",
        "PREV_APPLICATION_COUNT",
    ]

    with pytest.raises(AssertionError, match="duplicadas"):
        generator.validate_evidence(invalid)


def test_rejects_inconsistent_decision_flag(evidence):
    generator = load_generator()
    invalid = deepcopy(evidence)

    invalid["decision_strictly_before_reference"] = False

    with pytest.raises(AssertionError, match="DAYS_DECISION"):
        generator.validate_evidence(invalid)
