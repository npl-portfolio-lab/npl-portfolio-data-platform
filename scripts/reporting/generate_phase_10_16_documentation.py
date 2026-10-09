"""Genera documentación reproducible de la Fase 10.16."""

import json
from pathlib import Path

from scripts.ml.audit_pos_cash_temporal_sensitivity_v2 import audit


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "docs/ml"
BASENAME = "fase_10_16_sensibilidad_temporal_pos_cash_v2"

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
    "clients_without_history",
    "changed_record_count",
    "changed_max_dpd",
    "changed_avg_dpd",
    "changed_dpd_rate",
)


def validate_evidence(evidence: dict) -> None:
    if evidence.get("phase") != "10.16":
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

    if profile["records"] < 0 or profile["clients"] < 0:
        raise ValueError("Conteos negativos")

    if sensitivity["total_clients"] != profile["clients"]:
        raise ValueError("Total de clientes inconsistente")

    for field in (
        "clients_without_history",
        "changed_record_count",
        "changed_max_dpd",
        "changed_avg_dpd",
        "changed_dpd_rate",
    ):
        value = sensitivity[field]
        if not 0 <= value <= profile["clients"]:
            raise ValueError(f"Conteo fuera de rango: {field}")


def render_markdown(evidence: dict) -> str:
    validate_evidence(evidence)

    profile = evidence["temporal_profile"]
    sensitivity = evidence["sensitivity"]

    lines = [
        "# Fase 10.16 — Sensibilidad temporal POS_CASH_balance V2",
        "",
        "## Objetivo",
        "",
        "Evaluar la distribución temporal de POS_CASH_balance y "
        "medir la sensibilidad de algunas características al "
        "excluir el mes -1.",
        "",
        "## Fuente y metodología",
        "",
        "- Fuente: `data/processed/home_credit/POS_CASH_balance.parquet`",
        "- Referencia temporal: `MONTHS_BALANCE`.",
        "- Escenario original: todos los registros disponibles.",
        "- Escenario contrafactual: `MONTHS_BALANCE <= -2`.",
        "- Unidad de comparación: cliente (`SK_ID_CURR`).",
        "- Auditoría de solo lectura; no modifica ML V2.",
        "",
        "## Perfil temporal",
        "",
        "| Indicador | Valor |",
        "|---|---:|",
    ]

    labels_profile = {
        "records": "Registros",
        "clients": "Clientes",
        "contracts": "Contratos",
        "null_months": "Meses nulos",
        "nonhistorical_months": "Meses >= 0",
        "recent_records": "Registros del mes -1",
        "recent_clients": "Clientes del mes -1",
        "oldest_month": "Mes más antiguo",
        "newest_month": "Mes más reciente",
    }

    for field in PROFILE_FIELDS:
        lines.append(
            f"| {labels_profile[field]} | {profile[field]:,} |"
        )

    lines.extend([
        "",
        "## Sensibilidad al excluir el mes -1",
        "",
        "| Indicador | Clientes |",
        "|---|---:|",
    ])

    labels_sensitivity = {
        "total_clients": "Clientes originales",
        "clients_without_history": "Sin historial contrafactual",
        "changed_record_count": "POS_RECORD_COUNT diferente",
        "changed_max_dpd": "POS_MAX_DPD diferente",
        "changed_avg_dpd": "POS_AVG_DPD diferente",
        "changed_dpd_rate": "POS_DPD_RATE diferente",
    }

    for field in SENSITIVITY_FIELDS:
        lines.append(
            f"| {labels_sensitivity[field]} | "
            f"{sensitivity[field]:,} |"
        )

    lines.extend([
        "",
        "## Alcance y limitaciones",
        "",
        "- La comparación cubre cuatro métricas seleccionadas; "
        "no todas las características del constructor.",
        "- Las diferencias incluyen clientes sin historial "
        "en el escenario contrafactual.",
        "- `MONTHS_BALANCE < 0` indica meses anteriores a la "
        "referencia del dataset, pero no demuestra por sí solo "
        "disponibilidad operacional punto-en-tiempo.",
        "- La exclusión del mes -1 es un análisis de sensibilidad, "
        "no una corrección automática de fuga de información.",
        "- No se evalúa el impacto en predicciones ni métricas "
        "del modelo ML V2.",
        "",
        "## Conclusión",
        "",
        evidence["risk_classification"],
        "",
        "No se confirma Data Leakage con esta evidencia. "
        "Se requiere una validación adicional de disponibilidad "
        "punto-en-tiempo para afirmar ausencia de fuga.",
        "",
        "## Reproducción",
        "",
        "```bash",
        "python scripts/ml/audit_pos_cash_temporal_sensitivity_v2.py",
        "python -m scripts.reporting.generate_phase_10_16_documentation",
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
