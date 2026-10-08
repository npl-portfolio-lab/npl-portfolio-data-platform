"""Documentación automática de la Fase 10.7."""

import json

from npl_portfolio.core.paths import (
    PROJECT_ROOT,
    ML_ARTIFACTS_DIR,
)


REPORT = (
    ML_ARTIFACTS_DIR
    / "v2"
    / "logistic_models_comparison.json"
)

OUTPUT = (
    PROJECT_ROOT
    / "docs/ml/fase_10_7_entrenamiento_v2.md"
)

METRICS = (
    "roc_auc",
    "pr_auc",
    "precision",
    "recall",
    "f1",
)

COUNTS = ("tn", "fp", "fn", "tp")


def main():
    if not REPORT.is_file():
        raise FileNotFoundError(REPORT)

    data = json.loads(REPORT.read_text(encoding="utf-8"))

    if data.get("phase") != "10.7":
        raise ValueError("Fase incorrecta")

    if data.get("dataset_version") != "v2":
        raise ValueError("Versión incorrecta")

    if data.get("test_evaluated") is not False:
        raise ValueError("TEST debe permanecer reservado")

    models = data["models"]

    unbalanced = models["unbalanced"]
    balanced = models["balanced"]

    if data["features"] <= 0:
        raise ValueError("Número de variables inválido")

    for name, model in models.items():
        metrics = model["metrics"]

        if sum(metrics[key] for key in COUNTS) != data["validation_rows"]:
            raise ValueError(
                f"Matriz de confusión inconsistente: {name}"
            )

    metric_rows = []

    for metric in METRICS:
        left = unbalanced["metrics"][metric]
        right = balanced["metrics"][metric]

        metric_rows.append(
            f"| {metric.upper()} | {left:.6f} | {right:.6f} |"
        )

    confusion_rows = []

    for metric in COUNTS:
        left = unbalanced["metrics"][metric]
        right = balanced["metrics"][metric]

        confusion_rows.append(
            f"| {metric.upper()} | {left:,} | {right:,} |"
        )

    config = data["configuration"]

    content = f"""# Fase 10.7 — Entrenamiento y comparación V2

## 1. Objetivo

Entrenar y comparar regresiones logísticas con y sin
balanceo de clases sobre las particiones V2.

## 2. Datos utilizados

- TRAIN: {data["train_rows"]:,} registros.
- VALIDATION: {data["validation_rows"]:,} registros.
- Variables: {data["features"]}.
- TEST: reservado, sin evaluación predictiva.

## 3. Configuración experimental

- Escalador: {data["scaler"]}.
- Solver: {config["solver"]}.
- Iteraciones máximas: {config["max_iter"]}.
- Tolerancia: {config["tol"]}.
- Semilla: {config["random_state"]}.
- Umbral de clasificación: {config["threshold"]}.
- Modelo A: class_weight=None.
- Modelo B: class_weight=balanced.

El escalador se ajusta con TRAIN y se aplica a VALIDATION.

## 4. Métricas de VALIDATION

| Métrica | Sin balanceo | Balanceado |
|---|---:|---:|
{chr(10).join(metric_rows)}

## 5. Matrices de confusión

| Resultado | Sin balanceo | Balanceado |
|---|---:|---:|
{chr(10).join(confusion_rows)}

TN: verdadero negativo.
FP: falso positivo.
FN: falso negativo.
TP: verdadero positivo.

## 6. Entrenamiento

| Característica | Sin balanceo | Balanceado |
|---|---:|---:|
| Tiempo (segundos) | {unbalanced["training_seconds"]:.2f} | {balanced["training_seconds"]:.2f} |
| Iteraciones | {unbalanced["iterations"]} | {balanced["iterations"]} |
| Advertencia convergencia | {unbalanced["convergence_warning"]} | {balanced["convergence_warning"]} |

## 7. Interpretación

El modelo sin balanceo presenta mayor Precision al
umbral 0.50, mientras que el balanceado presenta
mayor Recall y F1.

Las diferencias de ROC-AUC y PR-AUC son pequeñas.
No se declara un ganador definitivo.

La elección del modelo dependerá también de la
capacidad operativa, los falsos positivos y Precision@K.

## 8. Auditoría y reproducibilidad

El script `scripts/ml/audit_logistic_models_v2.py`
permite verificar que los modelos guardados reproducen
las métricas del reporte.

Este documento se genera a partir del JSON de
resultados. No constituye por sí mismo evidencia
de ejecución satisfactoria de la auditoría.

## 9. Limitaciones

- TEST V2 no fue utilizado para evaluación predictiva.
- TEST V2 no es completamente inédito respecto a
  los experimentos históricos del proyecto.
- Sigue pendiente la auditoría temporal de variables
  para descartar fugas de información.
- Las métricas dependen del umbral de clasificación.
- No se ha evaluado todavía la estabilidad estadística
  de las diferencias entre los modelos V2.

## 10. Artefactos

- `data/artifacts/ml/v2/logistic_models_comparison.json`
- `data/artifacts/ml/v2/unbalanced_logistic_regression.joblib`
- `data/artifacts/ml/v2/balanced_logistic_regression.joblib`

## 11. Próxima fase

Evaluar la priorización operativa mediante Precision@K,
Recall@K y análisis de capacidad sobre VALIDATION V2.
"""

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT.write_text(content, encoding="utf-8")

    print("=" * 70)
    print("FASE 10.7 - DOCUMENTACION AUTOMATICA")
    print("=" * 70)
    print(f"[OK] TRAIN: {data['train_rows']:,}")
    print(f"[OK] VALIDATION: {data['validation_rows']:,}")
    print(f"[OK] Variables: {data['features']}")
    print(f"[OK] Documento: {OUTPUT}")


if __name__ == "__main__":
    main()
