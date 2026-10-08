"""Documentación automática de la Fase 10.8."""

import json

from npl_portfolio.core.paths import (
    PROJECT_ROOT,
    ML_ARTIFACTS_DIR,
)


ARTIFACT_DIR = ML_ARTIFACTS_DIR / "v2"
OUTPUT = PROJECT_ROOT / "docs/ml/fase_10_8_top_k_v2.md"


def main():
    report = json.loads(
        (ARTIFACT_DIR / "top_k_comparison.json").read_text(
            encoding="utf-8"
        )
    )
    overlap = json.loads(
        (ARTIFACT_DIR / "top_k_overlap.json").read_text(
            encoding="utf-8"
        )
    )

    assert report["phase"] == overlap["phase"] == "10.8"
    assert report["dataset_version"] == overlap["dataset_version"] == "v2"
    assert report["evaluation_partition"] == overlap["evaluation_partition"] == "validation"
    assert report["test_evaluated"] is False
    assert overlap["test_evaluated"] is False

    rows = []
    overlap_rows = []

    for k in report["k_values"]:
        key = str(k)

        assert key in overlap["overlap"]

        for name in ("unbalanced", "balanced"):
            metrics = report["models"][name][key]

            rows.append(
                f"| {k:,} | {name} | "
                f"{metrics['true_positives']:,} | "
                f"{metrics['precision_at_k']:.4f} | "
                f"{metrics['recall_at_k']:.4f} | "
                f"{metrics['lift_at_k']:.2f}x |"
            )

        item = overlap["overlap"][key]

        overlap_rows.append(
            f"| {k:,} | "
            f"{item['shared_clients']:,} | "
            f"{item['different_clients_per_model']:,} | "
            f"{item['overlap_percentage']:.2f}% | "
            f"{item['jaccard_similarity']:.4f} |"
        )

    content = f"""# Fase 10.8 — Evaluación operativa Top-K V2

## 1. Objetivo

Comparar la capacidad de priorización de los modelos
de regresión logística V2 sobre VALIDATION.

## 2. Datos utilizados

- Partición: VALIDATION V2.
- Registros: {report['validation_rows']:,}.
- Positivos: {report['positive_cases']:,}.
- Prevalencia: {report['prevalence']:.4%}.
- Variables: {report['features']}.
- TEST: no utilizado.

## 3. Metodología

Los clientes se ordenan por probabilidad descendente.

- Precision@K = TP@K / K.
- Recall@K = TP@K / total de positivos.
- Lift@K = Precision@K / prevalencia.

Los empates conservan el orden original de VALIDATION.

## 4. Resultados

| K | Modelo | TP@K | Precision@K | Recall@K | Lift@K |
|---|---|---:|---:|---:|---:|
{chr(10).join(rows)}

## 5. Coincidencia entre rankings

| K | Clientes compartidos | Diferentes por modelo | Coincidencia | Jaccard |
|---|---:|---:|---:|---:|
{chr(10).join(overlap_rows)}

La coincidencia representa la proporción de clientes
presentes en ambos conjuntos Top-K.

Jaccard es la intersección dividida entre la unión
de ambos conjuntos.

## 6. Interpretación

La comparación Top-K permite estudiar la priorización
operativa sin depender del umbral fijo de 0.50.

Una mayor Recall al umbral 0.50 no implica
necesariamente mejor priorización Top-K.

Las diferencias observadas son descriptivas.
No constituyen evidencia de superioridad estadística.

## 7. Limitaciones

- Evaluación realizada únicamente sobre VALIDATION V2.
- TEST permanece reservado.
- TEST V2 no es completamente inédito respecto a
  los experimentos históricos.
- La auditoría temporal de variables sigue pendiente.
- No se ha realizado inferencia estadística de
  diferencias Top-K.
- TARGET representa dificultad de pago, no
  recuperación efectiva de cartera castigada.

## 8. Artefactos

- `data/artifacts/ml/v2/top_k_comparison.json`
- `data/artifacts/ml/v2/top_k_overlap.json`

## 9. Reproducibilidad

- `scripts/ml/evaluate_top_k_v2.py`
- `scripts/ml/audit_top_k_v2.py`
- `tests/ml/test_top_k_evaluator.py`

Este documento se genera automáticamente desde
los resultados de la Fase 10.8.
"""

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(content, encoding="utf-8")

    print("=" * 70)
    print("FASE 10.8 - DOCUMENTACION AUTOMATICA")
    print("=" * 70)
    print(f"[OK] Registros: {report['validation_rows']:,}")
    print(f"[OK] Capacidades evaluadas: {report['k_values']}")
    print(f"[OK] Documento: {OUTPUT}")


if __name__ == "__main__":
    main()
