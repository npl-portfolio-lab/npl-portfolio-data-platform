"""Genera documentación Markdown de las Fases 10.1 y 10.2."""

import json
from pathlib import Path

from npl_portfolio.core.paths import DOCS_DIR, ML_ARTIFACTS_DIR


DOCS_OUTPUT = DOCS_DIR / "ml"

THRESHOLDS_PATH = (
    ML_ARTIFACTS_DIR / "advanced_threshold_evaluation.json"
)

CAPACITY_PATH = (
    ML_ARTIFACTS_DIR / "operational_capacity_analysis.json"
)


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(encoding="utf-8") as file:
        return json.load(file)


def generate_threshold_documentation(data: dict) -> str:
    lines = [
        "# Fase 10.1 — Evaluación avanzada de thresholds",
        "",
        "## Objetivo",
        "",
        "Analizar cómo cambia el desempeño de dos modelos de "
        "regresión logística al modificar el umbral de clasificación, "
        "sin realizar nuevos entrenamientos.",
        "",
        "## Datos de evaluación",
        "",
        f"- Dataset: `{data['dataset']}`",
        f"- Registros: {data['n_samples']:,}",
        f"- Variables: {data['n_features']:,}",
        f"- Positivos: {data['positive_samples']:,}",
        f"- Negativos: {data['negative_samples']:,}",
        "",
        "## Metodología",
        "",
        "Se reutilizaron los modelos entrenados y sus respectivos "
        "escaladores MaxAbsScaler. Para cada threshold se calcularon "
        "precision, recall, F1-score y matriz de confusión.",
        "",
        "Se verificó que ambos modelos reprodujeran sus matrices "
        "de confusión originales con threshold 0.50.",
        "",
        "## Resultados",
        "",
    ]

    for model_name, model_data in data["models"].items():
        lines.extend([
            f"### Modelo: {model_name}",
            "",
            "| Threshold | Precision | Recall | F1 | TP | FP | FN | Alertas |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])

        for result in model_data["results"]:
            lines.append(
                f"| {result['threshold']:.2f} "
                f"| {result['precision']:.4f} "
                f"| {result['recall']:.4f} "
                f"| {result['f1']:.4f} "
                f"| {result['tp']} "
                f"| {result['fp']} "
                f"| {result['fn']} "
                f"| {result['predicted_positives']} |"
            )

        best = max(
            model_data["results"],
            key=lambda result: result["f1"],
        )

        lines.extend([
            "",
            f"**Mejor F1 evaluado:** {best['f1']:.4f} "
            f"con threshold {best['threshold']:.2f}.",
            "",
        ])

    lines.extend([
        "## Interpretación",
        "",
        "El threshold 0.50 no necesariamente proporciona el mejor "
        "equilibrio entre precision y recall. Los modelos pueden "
        "necesitar thresholds diferentes porque producen "
        "distribuciones de probabilidades distintas.",
        "",
        "## Limitaciones",
        "",
        "- Los resultados corresponden al conjunto de validación.",
        "- Se evaluó una lista discreta de thresholds.",
        "- Maximizar F1 no equivale a optimizar el beneficio financiero.",
        "- TARGET=1 representa dificultades de pago, no recuperación "
        "monetaria de cartera castigada.",
        "- Se requiere una evaluación independiente antes de "
        "considerar una implementación productiva.",
        "",
        "## Artefacto",
        "",
        f"`data/artifacts/ml/{THRESHOLDS_PATH.name}`",
        "",
    ])

    return "\n".join(lines)


def generate_capacity_documentation(data: dict) -> str:
    lines = [
        "# Fase 10.2 — Análisis de capacidad operativa",
        "",
        "## Objetivo",
        "",
        "Seleccionar, entre las configuraciones evaluadas, "
        "el modelo y threshold que detecten más positivos reales "
        "sin superar un límite de alertas.",
        "",
        "## Datos",
        "",
        f"- Dataset: `{data['dataset']}`",
        f"- Registros: {data['n_samples']:,}",
        f"- Positivos reales: {data['positive_samples']:,}",
        "",
        "## Metodología",
        "",
        "Para cada capacidad máxima se seleccionó la configuración "
        "con más verdaderos positivos, siempre que las alertas "
        "no superaran el límite. En caso de empate se priorizó "
        "la configuración con menos falsos positivos.",
        "",
        "## Resultados",
        "",
        "| Capacidad | Modelo | Threshold | Alertas | TP | FP | "
        "Precision | Recall | Uso de capacidad |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for scenario in data["scenarios"]:
        capacity = scenario["maximum_alerts"]
        selected = scenario["selected"]

        if selected is None:
            lines.append(
                f"| {capacity:,} | Sin configuración viable "
                "| — | — | — | — | — | — | — |"
            )
            continue

        lines.append(
            f"| {capacity:,} "
            f"| {selected['model']} "
            f"| {selected['threshold']:.2f} "
            f"| {selected['alerts']:,} "
            f"| {selected['tp']:,} "
            f"| {selected['fp']:,} "
            f"| {selected['precision']:.4f} "
            f"| {selected['recall']:.4f} "
            f"| {selected['capacity_utilization']:.2%} |"
        )

    lines.extend([
        "",
        "## Interpretación",
        "",
        "El modelo seleccionado puede cambiar según la capacidad "
        "disponible. Una mayor capacidad permite revisar más "
        "registros, pero también puede incrementar los falsos positivos.",
        "",
        "## Limitaciones",
        "",
        "- Las capacidades representan alertas sobre el dataset "
        "completo de validación, no volúmenes diarios.",
        "- La selección utiliza únicamente los thresholds "
        "evaluados en la Fase 10.1.",
        "- Puede quedar capacidad sin utilizar.",
        "- Los resultados no incorporan costos de revisión "
        "ni pérdidas económicas por errores.",
        "- No representan una validación independiente en producción.",
        "",
        "## Próxima etapa",
        "",
        "Fase 10.3: evaluación Top-K para comparar modelos "
        "utilizando exactamente la misma cantidad de alertas.",
        "",
        "## Artefacto",
        "",
        f"`data/artifacts/ml/{CAPACITY_PATH.name}`",
        "",
    ])

    return "\n".join(lines)


def main() -> None:
    thresholds = load_json(THRESHOLDS_PATH)
    capacity = load_json(CAPACITY_PATH)

    if thresholds.get("phase") != "10.1":
        raise ValueError("Reporte de thresholds incompatible")

    if capacity.get("phase") != "10.2":
        raise ValueError("Reporte de capacidad incompatible")

    documents = {
        "fase_10_1_thresholds.md": (
            generate_threshold_documentation(thresholds)
        ),
        "fase_10_2_capacidad_operativa.md": (
            generate_capacity_documentation(capacity)
        ),
    }

    DOCS_OUTPUT.mkdir(parents=True, exist_ok=True)

    for filename, content in documents.items():
        path = DOCS_OUTPUT / filename
        path.write_text(content, encoding="utf-8")
        print(f"[OK] Documentación generada: {path}")

    print("[OK] Documentación Fase 10.1 y 10.2 finalizada")


if __name__ == "__main__":
    main()
