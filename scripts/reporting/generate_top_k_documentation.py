"""Genera documentación técnica de la Fase 10.3."""

import json

from npl_portfolio.core.paths import DOCS_DIR, ML_ARTIFACTS_DIR


INPUT_PATH = ML_ARTIFACTS_DIR / "top_k_evaluation.json"
OUTPUT_PATH = DOCS_DIR / "ml" / "fase_10_3_top_k.md"


def main():
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(INPUT_PATH)

    with INPUT_PATH.open(encoding="utf-8") as file:
        data = json.load(file)

    if data.get("phase") != "10.3":
        raise ValueError("Reporte incompatible con Fase 10.3")

    lines = [
        "# Fase 10.3 — Evaluación Top-K",
        "",
        "## Objetivo",
        "",
        "Comparar la capacidad de priorización de dos modelos de "
        "regresión logística utilizando el mismo número de alertas.",
        "",
        "## Datos",
        "",
        f"- Dataset: `{data['dataset']}`",
        f"- Registros: {data['n_samples']:,}",
        f"- Variables: {data['n_features']:,}",
        f"- Positivos: {data['positive_samples']:,}",
        f"- Negativos: {data['negative_samples']:,}",
        f"- Prevalencia: {data['prevalence']:.2%}",
        "",
        "## Metodología",
        "",
        "Se cargaron ambos modelos y sus escaladores previamente "
        "entrenados. Los registros se ordenaron de mayor a menor "
        "probabilidad estimada de dificultad de pago.",
        "",
        "Se seleccionaron exactamente los primeros K registros "
        "para cada capacidad. Los empates se resolvieron "
        "manteniendo el orden original del dataset.",
        "",
        "Las métricas utilizadas fueron:",
        "",
        "- Precision@K = TP / K.",
        "- Recall@K = TP / total de positivos reales.",
        "- Lift@K = Precision@K / prevalencia.",
        "",
        "## Resultados",
        "",
        "| Modelo | K | TP | FP | Precision@K | Recall@K | Lift@K |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for name, model in data["models"].items():
        for result in model["results"]:
            lines.append(
                f"| {name} "
                f"| {result['k']:,} "
                f"| {result['tp']:,} "
                f"| {result['fp']:,} "
                f"| {result['precision_at_k']:.4f} "
                f"| {result['recall_at_k']:.4f} "
                f"| {result['lift_at_k']:.4f} |"
            )

    lines.extend([
        "",
        "## Interpretación",
        "",
        "Top-K permite comparar ambos modelos bajo una capacidad "
        "operativa idéntica, sin depender de thresholds fijos.",
        "",
        "Un Lift@K superior a 1 indica una concentración de "
        "positivos mayor que la prevalencia del dataset.",
        "",
        "## Limitaciones",
        "",
        "- Los resultados corresponden exclusivamente a VALIDATION.",
        "- Las capacidades representan registros del conjunto "
        "completo, no alertas diarias.",
        "- La evaluación no incorpora costos financieros.",
        "- TARGET=1 representa dificultad de pago, no recuperación "
        "monetaria de cartera.",
        "- La elección definitiva requiere evaluación independiente.",
        "- Los empates en probabilidad pueden afectar qué registros "
        "quedan en el límite de K.",
        "",
        "## Artefactos",
        "",
        "- `data/artifacts/ml/top_k_evaluation.json`",
        "- `scripts/ml/evaluate_top_k.py`",
        "",
    ])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(f"[OK] Documentación generada: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
