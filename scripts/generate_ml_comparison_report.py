from pathlib import Path
from datetime import datetime, timezone
import json

ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "data" / "artifacts" / "ml" / "logistic_models_comparison.json"

OUTPUT = ROOT / "docs" / "machine_learning" / "phase_9_6_controlled_comparison.md"


def percentage(value):
    return f"{value * 100:.2f} %"


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(f"No existe el archivo de resultados: {SOURCE}")

    with SOURCE.open(encoding="utf-8") as file:
        data = json.load(file)

    unbalanced = data["unbalanced"]
    balanced = data["balanced"]

    a = unbalanced["metrics"]
    b = balanced["metrics"]

    rows = [
        ("ROC-AUC", a["roc_auc"], b["roc_auc"]),
        ("PR-AUC", a["pr_auc"], b["pr_auc"]),
        ("Precision", a["precision"], b["precision"]),
        ("Recall", a["recall"], b["recall"]),
        ("F1-score", a["f1"], b["f1"]),
    ]

    metrics_table = "\n".join(
        f"| {name} | {percentage(x)} | {percentage(y)} |" for name, x, y in rows
    )

    confusion_table = "\n".join(
        f"| {name} | {a[key]:,} | {b[key]:,} |"
        for name, key in [
            ("Verdaderos negativos (TN)", "tn"),
            ("Falsos positivos (FP)", "fp"),
            ("Falsos negativos (FN)", "fn"),
            ("Verdaderos positivos (TP)", "tp"),
        ]
    )

    fn_reduction = a["fn"] - b["fn"]
    fp_increase = b["fp"] - a["fp"]
    tp_increase = b["tp"] - a["tp"]

    config = data["configuration"]

    content = f"""# Fase 9.6 — Comparación controlada de modelos

**Proyecto:** NPL Portfolio Data Platform  
**Fecha de generación:** {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}  
**Estado:** Experimento completado

## 1. Objetivo

Evaluar el efecto del balanceo de clases en una regresión logística
que predice dificultades de pago (`TARGET = 1`).

La comparación utiliza el mismo conjunto de entrenamiento,
el mismo conjunto de validación y la misma configuración técnica.

La única diferencia experimental es el parámetro `class_weight`.

## 2. Configuración del experimento

| Parámetro | Valor |
|---|---|
| Algoritmo | LogisticRegression |
| Escalador | {config["scaler"]} |
| Solver | {config["solver"]} |
| max_iter | {config["max_iter"]} |
| tol | {config["tol"]} |
| random_state | {config["random_state"]} |
| Threshold | {config["threshold"]} |
| Modelo A | class_weight=None |
| Modelo B | class_weight=balanced |

El escalador se ajusta exclusivamente con TRAIN y posteriormente
se aplica a VALIDATION para evitar fuga de información.

## 3. Resultados de evaluación

| Métrica | Sin balanceo | Balanceado |
|---|---:|---:|
{metrics_table}

## 4. Matriz de confusión

| Resultado | Sin balanceo | Balanceado |
|---|---:|---:|
{confusion_table}

## 5. Interpretación

### Modelo sin balanceo

- Precision: {percentage(a["precision"])}.
- Recall: {percentage(a["recall"])}.
- Identifica correctamente {a["tp"]:,} casos positivos.
- No identifica {a["fn"]:,} casos positivos.
- Genera {a["fp"]:,} falsos positivos.

Este modelo es conservador al clasificar clientes como positivos.
Aunque sus predicciones positivas son relativamente precisas,
detecta una proporción muy pequeña de los casos reales.

### Modelo balanceado

- Precision: {percentage(b["precision"])}.
- Recall: {percentage(b["recall"])}.
- Identifica correctamente {b["tp"]:,} casos positivos.
- No identifica {b["fn"]:,} casos positivos.
- Genera {b["fp"]:,} falsos positivos.

Este modelo detecta una proporción considerablemente mayor
de clientes con dificultades de pago, a costa de incrementar
las alertas incorrectas.

## 6. Diferencias principales

Al utilizar balanceo de clases:

- Se detectan {tp_increase:,} casos positivos adicionales.
- Se reducen los falsos negativos en {fn_reduction:,}.
- Se generan {fp_increase:,} falsos positivos adicionales.
- El Recall aumenta de {percentage(a["recall"])}
  a {percentage(b["recall"])}.
- El F1-score aumenta de {percentage(a["f1"])}
  a {percentage(b["f1"])}.

Las diferencias en ROC-AUC y PR-AUC son pequeñas.
El cambio más importante aparece en las decisiones
de clasificación al utilizar el umbral 0.50.

## 7. Conclusiones

El balanceo de clases mejora significativamente la capacidad
de detección de la clase positiva al umbral evaluado.

Sin embargo, esta mejora implica un incremento considerable
de falsos positivos, que podrían traducirse en revisiones
innecesarias o decisiones de riesgo incorrectas.

Por tanto, no se selecciona todavía un modelo definitivo
para su utilización operativa.

Ambos modelos deben considerarse en la siguiente fase,
donde se evaluarán diferentes umbrales de decisión,
el comportamiento de los errores y sus posibles
implicaciones para el negocio.

## 8. Próxima fase

**Fase 10 — Evaluación avanzada y análisis de umbrales.**

Se analizará el compromiso entre Precision y Recall,
la cantidad de falsos positivos y falsos negativos
y los criterios para seleccionar un umbral adecuado.

---

*Documento generado automáticamente a partir de los resultados
del experimento de comparación controlada.*
"""

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    temp_path = OUTPUT.with_suffix(".md.tmp")
    temp_path.write_text(content, encoding="utf-8")
    temp_path.replace(OUTPUT)

    print(f"[OK] Documento generado: {OUTPUT}")


if __name__ == "__main__":
    main()
