"""Documentación reproducible de la Fase 10.17."""

import json
from pathlib import Path

from scripts.ml.audit_credit_card_temporal_sensitivity_v2 import audit


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "docs/ml"
BASENAME = "fase_10_17_sensibilidad_temporal_credit_card_v2"

PROFILE_FIELDS = (
    "records",
    "clients",
    "contracts",
    "null_months",
    "nonhistorical_months",
    "recent_records",
    "recent_clients",
    "oldest_month",
    "newest_month",
)

SENSITIVITY_FIELDS = (
    "total_clients",
    "comparable_clients",
    "clients_without_history",
    "changed_record_count",
    "changed_max_dpd",
    "changed_avg_dpd",
    "changed_dpd_rate",
    "changed_avg_balance",
    "changed_max_balance",
    "changed_avg_credit_limit",
    "changed_avg_utilization",
    "changed_total_payment",
)


def validate_evidence(evidence: dict) -> None:
    if evidence.get("phase") != "10.17":
        raise ValueError("Fase incorrecta")

    if evidence.get("cutoff_month") != -2:
        raise ValueError("Corte temporal incorrecto")

    profile = evidence.get("temporal_profile")
    sensitivity = evidence.get("sensitivity")

    if not isinstance(profile, dict):
        raise ValueError("Perfil temporal inválido")

    if not isinstance(sensitivity, dict):
        raise ValueError("Sensibilidad inválida")

    for field in PROFILE_FIELDS:
        if field not in profile:
            raise ValueError(f"Falta campo temporal: {field}")

    for field in SENSITIVITY_FIELDS:
        if field not in sensitivity:
            raise ValueError(f"Falta campo de sensibilidad: {field}")

    if (
        sensitivity["total_clients"]
        != sensitivity["comparable_clients"]
        + sensitivity["clients_without_history"]
    ):
        raise ValueError("Conteos de clientes inconsistentes")

    if sensitivity["total_clients"] != profile["clients"]:
        raise ValueError("Total de clientes inconsistente")

    comparable = sensitivity["comparable_clients"]

    for field in SENSITIVITY_FIELDS:
        value = sensitivity[field]

        if not isinstance(value, int) or value < 0:
            raise ValueError(f"Conteo inválido: {field}")

        if field.startswith("changed_") and value > comparable:
            raise ValueError(f"Cambios fuera de rango: {field}")


def render_markdown(evidence: dict) -> str:
    validate_evidence(evidence)

    profile = evidence["temporal_profile"]
    sensitivity = evidence["sensitivity"]

    lines = [
        "# Fase 10.17 — Sensibilidad temporal credit_card_balance V2",
        "",
        "## Objetivo",
        "",
        "Evaluar el historial temporal de tarjetas de crédito "
        "y medir la sensibilidad de características seleccionadas "
        "al excluir el mes -1.",
        "",
        "## Metodología",
        "",
        "- Fuente: `credit_card_balance.parquet`.",
        "- Escenario original: historial completo.",
        "- Escenario contrafactual: `MONTHS_BALANCE <= -2`.",
        "- Comparación de características: solo clientes "
        "presentes en ambos escenarios.",
        "- Clientes sin historial contrafactual: reportados "
        "por separado.",
        "- DuckDB configurado con un hilo para reproducibilidad.",
        "- Análisis de solo lectura; ML V2 permanece intacto.",
        "",
        "## Perfil temporal",
        "",
        "| Indicador | Valor |",
        "|---|---:|",
    ]

    for field in PROFILE_FIELDS:
        lines.append(f"| {field} | {profile[field]:,} |")

    lines.extend([
        "",
        "## Sensibilidad temporal",
        "",
        "| Indicador | Clientes |",
        "|---|---:|",
    ])

    for field in SENSITIVITY_FIELDS:
        lines.append(
            f"| {field} | {sensitivity[field]:,} |"
        )

    lines.extend([
        "",
        "## Interpretación",
        "",
        "Los conteos `changed_*` representan clientes "
        "comparables con diferencias entre los dos escenarios.",
        "",
        "La exclusión del mes -1 es un experimento de "
        "sensibilidad, no una corrección automática de leakage.",
        "",
        "## Limitaciones",
        "",
    ])

    for limitation in evidence.get("limitations", []):
        lines.append(f"- {limitation}")

    lines.extend([
        "",
        "## Conclusión",
        "",
        evidence["risk_classification"],
        "",
        "La ausencia de meses posteriores a la referencia "
        "no demuestra por sí sola disponibilidad punto-en-tiempo. "
        "Los cambios observados no prueban Data Leakage.",
        "",
        "## Reproducción",
        "",
        "```bash",
        "python scripts/ml/audit_credit_card_temporal_sensitivity_v2.py",
        "python -m scripts.reporting.generate_phase_10_17_documentation",
        "python -m pytest -q",
        "```",
        "",
    ])

    return "\n".join(lines)


def generate(
    evidence: dict | None = None,
    output_dir: Path = OUTPUT_DIR,
) -> tuple[Path, Path]:
    if evidence is None:
        evidence = audit()

    validate_evidence(evidence)
    markdown = render_markdown(evidence)

    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{BASENAME}.json"
    markdown_path = output_dir / f"{BASENAME}.md"

    json_path.write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(markdown, encoding="utf-8")

    return json_path, markdown_path


if __name__ == "__main__":
    for path in generate():
        print(f"Generado: {path.relative_to(ROOT)}")
