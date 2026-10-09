"""Pruebas de documentación de la Fase 10.17."""

import json

import pytest

from scripts.reporting.generate_phase_10_17_documentation import (
    generate,
    render_markdown,
    validate_evidence,
)


@pytest.fixture
def evidence():
    return {
        "phase": "10.17",
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
            "comparable_clients": 2,
            "clients_without_history": 1,
            "changed_record_count": 2,
            "changed_max_dpd": 1,
            "changed_avg_dpd": 1,
            "changed_dpd_rate": 1,
            "changed_avg_balance": 1,
            "changed_max_balance": 1,
            "changed_avg_credit_limit": 0,
            "changed_avg_utilization": 1,
            "changed_total_payment": 2,
        },
        "risk_classification": "Data Leakage no confirmado",
        "limitations": [
            "No demuestra disponibilidad punto-en-tiempo."
        ],
    }


def test_valid_evidence(evidence):
    validate_evidence(evidence)


def test_rejects_missing_field(evidence):
    del evidence["sensitivity"]["comparable_clients"]

    with pytest.raises(ValueError, match="Falta campo"):
        validate_evidence(evidence)


def test_rejects_inconsistent_clients(evidence):
    evidence["sensitivity"]["comparable_clients"] = 1

    with pytest.raises(ValueError, match="inconsistentes"):
        validate_evidence(evidence)


def test_markdown_contains_findings(evidence):
    content = render_markdown(evidence)

    assert "Fase 10.17" in content
    assert "credit_card_balance" in content
    assert "changed_avg_utilization" in content
    assert "Data Leakage no confirmado" in content


def test_generate_json_and_markdown(evidence, tmp_path):
    json_path, md_path = generate(evidence, tmp_path)

    assert json_path.exists()
    assert md_path.exists()

    saved = json.loads(json_path.read_text(encoding="utf-8"))

    assert saved == evidence
    assert "Sensibilidad temporal" in md_path.read_text(
        encoding="utf-8"
    )
