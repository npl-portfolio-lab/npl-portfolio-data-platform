# Fase 10.3 — Evaluación Top-K

## Objetivo

Comparar la capacidad de priorización de dos modelos de regresión logística utilizando el mismo número de alertas.

## Datos

- Dataset: `validation`
- Registros: 61,503
- Variables: 378
- Positivos: 4,965
- Negativos: 56,538
- Prevalencia: 8.07%

## Metodología

Se cargaron ambos modelos y sus escaladores previamente entrenados. Los registros se ordenaron de mayor a menor probabilidad estimada de dificultad de pago.

Se seleccionaron exactamente los primeros K registros para cada capacidad. Los empates se resolvieron manteniendo el orden original del dataset.

Las métricas utilizadas fueron:

- Precision@K = TP / K.
- Recall@K = TP / total de positivos reales.
- Lift@K = Precision@K / prevalencia.

## Resultados

| Modelo | K | TP | FP | Precision@K | Recall@K | Lift@K |
|---|---:|---:|---:|---:|---:|---:|
| unbalanced | 1,000 | 445 | 555 | 0.4450 | 0.0896 | 5.5124 |
| unbalanced | 3,000 | 1,076 | 1,924 | 0.3587 | 0.2167 | 4.4429 |
| unbalanced | 5,000 | 1,527 | 3,473 | 0.3054 | 0.3076 | 3.7831 |
| balanced | 1,000 | 447 | 553 | 0.4470 | 0.0900 | 5.5371 |
| balanced | 3,000 | 1,062 | 1,938 | 0.3540 | 0.2139 | 4.3851 |
| balanced | 5,000 | 1,523 | 3,477 | 0.3046 | 0.3067 | 3.7732 |

## Interpretación

Top-K permite comparar ambos modelos bajo una capacidad operativa idéntica, sin depender de thresholds fijos.

Un Lift@K superior a 1 indica una concentración de positivos mayor que la prevalencia del dataset.

## Limitaciones

- Los resultados corresponden exclusivamente a VALIDATION.
- Las capacidades representan registros del conjunto completo, no alertas diarias.
- La evaluación no incorpora costos financieros.
- TARGET=1 representa dificultad de pago, no recuperación monetaria de cartera.
- La elección definitiva requiere evaluación independiente.
- Los empates en probabilidad pueden afectar qué registros quedan en el límite de K.

## Artefactos

- `data/artifacts/ml/top_k_evaluation.json`
- `scripts/ml/evaluate_top_k.py`
