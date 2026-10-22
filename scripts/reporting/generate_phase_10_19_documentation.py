"""Genera la matriz consolidada de riesgo temporal ML V2."""

from pathlib import Path

from npl_portfolio.core.paths import PROJECT_ROOT
from scripts.ml.audit_consolidated_temporal_risk_v2 import audit, consolidate, get_contract_feature_distribution


OUTPUT = (
    PROJECT_ROOT
    / "docs/ml/fase_10_19_matriz_riesgo_temporal_v2.md"
)


def format_value(value):
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def render_markdown(result: dict) -> str:
    if result.get("phase") != "10.19":
        raise ValueError("Fase consolidada incorrecta")

    rows = result.get("risk_matrix", [])

    if len(rows) != 6:
        raise ValueError("Se requieren seis fuentes auditadas")

    lines = [
        "# Fase 10.19 — Matriz consolidada de riesgo temporal ML V2",
        "",
        "## Objetivo",
        "",
        "Consolidar la evidencia de las auditorías temporales "
        "10.13 a 10.18 sin modificar el pipeline ML V2.",
        "",
        "## Matriz de fuentes",
        "",
        "| Fuente | Fase | Clasificación |",
        "|---|---|---|",
    ]

    for row in rows:
        classification = str(row["classification"]).replace("|", "/")
        lines.append(
            f'| {row["label"]} | {row["phase"]} | '
            f'{classification} |'
        )

    lines.extend([
        "",
        "## Distribución de variables del contrato ML V2",
        "",
        "| Fuente | Variables numéricas |",
        "|---|---:|",
    ])

    distribution = result.get("feature_distribution")

    if distribution is not None:
        labels = {
            "bureau": "Bureau",
            "bureau_balance": "Bureau Balance",
            "previous_application": "Previous Application",
            "pos_cash": "POS CASH",
            "credit_card": "Credit Card",
            "installments": "Installments",
            "application": "Application",
        }

        for source, count in distribution["distribution"].items():
            lines.append(f"| {labels[source]} | {count:,} |")

        lines.extend([
            f'| **Total** | **{distribution["total_numeric_features"]:,}** |',
            "",
            f'Entradas categóricas: {distribution["total_categorical_inputs"]:,}.',
            "",
        ])

    lines.extend([
        "## Evidencia por fuente",
        "",
    ])

    for row in rows:
        lines.extend([
            f'### {row["label"]} — Fase {row["phase"]}',
            "",
        ])

        evidence = row["evidence"]

        for key, value in evidence.items():
            if key in {"phase", "source", "risk_classification"}:
                continue

            if isinstance(value, dict):
                lines.extend([
                    f"#### {key}",
                    "",
                    "| Indicador | Resultado |",
                    "|---|---:|",
                ])

                for field, field_value in value.items():
                    if isinstance(field_value, dict):
                        lines.extend([
                            "",
                            f"##### {field}",
                            "",
                            "| Característica | Resultado |",
                            "|---|---:|",
                        ])

                        for feature, count in field_value.items():
                            lines.append(
                                f"| `{feature}` | "
                                f"{format_value(count)} |"
                            )
                    elif not isinstance(field_value, (list, dict)):
                        lines.append(
                            f"| `{field}` | "
                            f"{format_value(field_value)} |"
                        )

                lines.append("")

            elif isinstance(value, (int, float, str, bool)):
                lines.append(
                    f"- **{key}:** {format_value(value)}"
                )

        lines.append("")

    lines.extend([
        "## Interpretación consolidada",
        "",
        result["conclusion"],
        "",
        "## Integridad del análisis",
        "",
        "- Evidencia obtenida de los auditores existentes.",
        "- No se modifican datasets ni modelos ML V2.",
        "- No se reentrenan modelos.",
        "- No se publican archivos JSON.",
        "- La sensibilidad no equivale a leakage confirmado.",
        "",
        "## Reproducción",
        "",
        "```bash",
        "python -m scripts.reporting.generate_phase_10_19_documentation",
        "```",
        "",
    ])

    return "\n".join(lines)


def generate(evidence=None, output: Path = OUTPUT) -> Path:
    result = audit() if evidence is None else consolidate(evidence)

    result['feature_distribution'] = get_contract_feature_distribution()

    content = render_markdown(result)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")

    return output


if __name__ == "__main__":
    print(generate())
