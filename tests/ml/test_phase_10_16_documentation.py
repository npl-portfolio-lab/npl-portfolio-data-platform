"""Pruebas de documentación de la Fase 10.16."""

import json

import pytest

from scripts.reporting.generate_phase_10_16_documentation import (
    generate,
    render_markdown,
    validate_evidence,
)


@pytest.fixture
def evidence():
    return {
        "phase": "10.16",
        "source": "test.parquet",
        "cutoff_month": -2,
        "temporal_profile": {
            "records": 5,
            "clients": 3,
            "contracts": 3,
            "null_months": 0,
            "nonhistorical_months": 0,
            "recent_records": 3,
            "recent_clients": 3,
            "oldest_month": -3,
            "newest_month": -1,
        },
        "sensitivity": {
            "total_clients": 3,
            "clients_without_history": 1,
            "changed_record_count": 3,
            "changed_max_dpd": 2,
            "changed_avg_dpd": 2,
            "changed_dpd_rate": 2,
        },
        "risk_classification": "Data Leakage no confirmado",
    }


def test_valid_evidence(evidence):
    validate_evidence(evidence)


def test_rejects_missing_field(evidence):
    del evidence["temporal_profile"]["records"]

    with pytest.raises(ValueError, match="Falta campo"):
        validate_evidence(evidence)


def test_rejects_inconsistent_clients(evidence):
    evidence["sensitivity"]["total_clients"] = 99

    with pytest.raises(ValueError, match="inconsistente"):
        validate_evidence(evidence)


def test_markdown_contains_findings(evidence):
    content = render_markdown(evidence)

    assert "Fase 10.16" in content
    assert "POS_CASH_balance" in content
    assert "POS_AVG_DPD" in content
    assert "Data Leakage no confirmado" in content


def test_generate_json_and_markdown(evidence, tmp_path):
    json_path, md_path = generate(evidence, tmp_path)

    assert json_path.exists()
    assert md_path.exists()

    saved = json.loads(json_path.read_text(encoding="utf-8"))

    assert saved == evidence
    assert "Sensibilidad" in md_path.read_text(encoding="utf-8")
