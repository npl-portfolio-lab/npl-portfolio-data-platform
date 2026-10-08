"""Documentación técnica de la Fase 10.4."""

import json

from npl_portfolio.core.paths import DOCS_DIR, ML_ARTIFACTS_DIR


INPUT_PATH = ML_ARTIFACTS_DIR / "bootstrap_stability_evaluation.json"
OUTPUT_PATH = DOCS_DIR / "ml" / "fase_10_4_bootstrap.md"


def main():
    with INPUT_PATH.open(encoding="utf-8") as file:
        data = json.load(file)

    if data.get("phase") != "10.4":
        raise ValueError("Reporte incompatible")

    lines = [
        "# Fase 10.4 — Estabilidad estadística mediante bootstrap pareado",
        "",
        "## Objetivo",
        "",
        "Estimar la incertidumbre de las diferencias entre los "
        "modelos sin balanceo y balanceado en la priorización Top-K.",
        "",
        "## Configuración",
        "",
        f"- Dataset: `{data['dataset']}`",
        f"- Registros: {data['n_samples']:,}",
        f"- Repeticiones: {data['n_bootstrap']:,}",
        f"- Semilla aleatoria: {data['random_state']}",
        f"- Nivel de confianza: {data['confidence_level']:.0%}",
        f"- Método: `{data['method']}`",
        "- Diferencia: balanceado menos sin balanceo.",
        "",
        "## Metodología",
        "",
        "En cada repetición se generó una muestra con reemplazo "
        "del conjunto VALIDATION. Ambos modelos se evaluaron "
        "sobre los mismos índices, recalculando el ranking Top-K.",
        "",
        "Los intervalos se calcularon mediante percentiles "
        "2.5 % y 97.5 % de las diferencias bootstrap.",
        "",
        "## Resultados",
        "",
        "| K | Diferencia media Precision@K | IC 95 % inferior "
        "| IC 95 % superior | Excluye cero |",
        "|---:|---:|---:|---:|---|",
    ]

    for k in data["capacities"]:
        result = data["results"][str(k)]

        diff = result["difference_precision"]
        ci = diff["confidence_interval"]

        lines.append(
            f"| {k:,} "
            f"| {diff['mean']:.6f} "
            f"| {ci['lower']:.6f} "
            f"| {ci['upper']:.6f} "
            f"| {'Sí' if result['precision_difference_excludes_zero'] else 'No'} |"
        )

    lines.extend([
        "",
        "## Interpretación",
        "",
        "Si el intervalo de confianza de la diferencia incluye "
        "el cero, los resultados no aportan evidencia suficiente "
        "para afirmar una diferencia consistente bajo este método.",
        "",
        "Un intervalo que excluya cero sugiere una diferencia "
        "sistemática en las remuestras, aunque no garantiza "
        "superioridad futura ni utilidad económica.",
        "",
        "## Limitaciones",
        "",
        "- Bootstrap estima incertidumbre condicionada a VALIDATION.",
        "- No sustituye una evaluación en TEST independiente.",
        "- Los modelos se mantienen fijos; no se reentrenan.",
        "- Se asume que las observaciones pueden remuestrearse "
        "de manera independiente.",
        "- Los intervalos percentiles pueden tener limitaciones "
        "en muestras sesgadas o con dependencia temporal.",
        "- Los empates de puntuación se resuelven mediante orden estable.",
        "- No se incorporan costos financieros.",
        "",
        "## Artefactos",
        "",
        "- `data/artifacts/ml/bootstrap_stability_evaluation.json`",
        "- `scripts/ml/evaluate_bootstrap_stability.py`",
        "",
    ])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print(f"[OK] Documentación generada: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
