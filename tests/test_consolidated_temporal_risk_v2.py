"""Pruebas de consolidación temporal ML V2."""

import pytest

from scripts.ml.audit_consolidated_temporal_risk_v2 import (
    EXPECTED_PHASES,
    consolidate,
)
from scripts.reporting.generate_phase_10_19_documentation import (
    generate,
    render_markdown,
)


@pytest.fixture
def evidence():
    return {
        source: {
            "phase": phase,
            "risk_classification": "Revisión temporal pendiente",
            "temporal_profile": {
                "records": 100,
                "clients": 20,
            },
            "sensitivity": {
                "changed_record_count": 5,
            },
        }
        for source, phase in EXPECTED_PHASES.items()
    }


def test_consolidates_six_sources(evidence):
    result = consolidate(evidence)

    assert result["phase"] == "10.19"
    assert result["sources_evaluated"] == 6
    assert len(result["risk_matrix"]) == 6


def test_rejects_missing_source(evidence):
    evidence.pop("bureau")

    with pytest.raises(ValueError, match="bureau"):
        consolidate(evidence)


def test_rejects_wrong_phase(evidence):
    evidence["pos_cash"]["phase"] = "10.99"

    with pytest.raises(ValueError, match="Fase incorrecta"):
        consolidate(evidence)


def test_renders_markdown(evidence):
    markdown = render_markdown(consolidate(evidence))

    assert "Fase 10.19" in markdown
    assert "Bureau Balance" in markdown
    assert "Previous Application" in markdown
    assert "Installments" in markdown


def test_generates_markdown_without_running_auditors(
    evidence, tmp_path
):
    output = tmp_path / "fase_10_19.md"

    generated = generate(evidence=evidence, output=output)

    assert generated == output
    assert output.is_file()
    assert "Fase 10.19" in output.read_text(
        encoding="utf-8"
    )
