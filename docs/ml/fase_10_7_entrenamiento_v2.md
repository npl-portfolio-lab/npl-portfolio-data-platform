# Fase 10.7 — Entrenamiento y comparación V2

## 1. Objetivo

Entrenar y comparar regresiones logísticas con y sin
balanceo de clases sobre las particiones V2.

## 2. Datos utilizados

- TRAIN: 196,806 registros.
- VALIDATION: 49,202 registros.
- Variables: 378.
- TEST: reservado, sin evaluación predictiva.

## 3. Configuración experimental

- Escalador: MaxAbsScaler.
- Solver: liblinear.
- Iteraciones máximas: 200.
- Tolerancia: 0.001.
- Semilla: 42.
- Umbral de clasificación: 0.5.
- Modelo A: class_weight=None.
- Modelo B: class_weight=balanced.

El escalador se ajusta con TRAIN y se aplica a VALIDATION.

## 4. Métricas de VALIDATION

| Métrica | Sin balanceo | Balanceado |
|---|---:|---:|
| ROC_AUC | 0.768303 | 0.768716 |
| PR_AUC | 0.243756 | 0.242483 |
| PRECISION | 0.464286 | 0.172560 |
| RECALL | 0.026183 | 0.695368 |
| F1 | 0.049571 | 0.276504 |

## 5. Matrices de confusión

| Resultado | Sin balanceo | Balanceado |
|---|---:|---:|
| TN | 45,110 | 31,986 |
| FP | 120 | 13,244 |
| FN | 3,868 | 1,210 |
| TP | 104 | 2,762 |

TN: verdadero negativo.
FP: falso positivo.
FN: falso negativo.
TP: verdadero positivo.

## 6. Entrenamiento

| Característica | Sin balanceo | Balanceado |
|---|---:|---:|
| Tiempo (segundos) | 13.48 | 29.20 |
| Iteraciones | 17 | 9 |
| Advertencia convergencia | False | False |

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
