# Fase 10.8 — Evaluación operativa Top-K V2

## 1. Objetivo

Comparar la capacidad de priorización de los modelos
de regresión logística V2 sobre VALIDATION.

## 2. Datos utilizados

- Partición: VALIDATION V2.
- Registros: 49,202.
- Positivos: 3,972.
- Prevalencia: 8.0728%.
- Variables: 378.
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
| 1,000 | unbalanced | 402 | 0.4020 | 0.1012 | 4.98x |
| 1,000 | balanced | 393 | 0.3930 | 0.0989 | 4.87x |
| 3,000 | unbalanced | 944 | 0.3147 | 0.2377 | 3.90x |
| 3,000 | balanced | 946 | 0.3153 | 0.2382 | 3.91x |
| 5,000 | unbalanced | 1,365 | 0.2730 | 0.3437 | 3.38x |
| 5,000 | balanced | 1,364 | 0.2728 | 0.3434 | 3.38x |

## 5. Coincidencia entre rankings

| K | Clientes compartidos | Diferentes por modelo | Coincidencia | Jaccard |
|---|---:|---:|---:|---:|
| 1,000 | 938 | 62 | 93.80% | 0.8832 |
| 3,000 | 2,838 | 162 | 94.60% | 0.8975 |
| 5,000 | 4,765 | 235 | 95.30% | 0.9102 |

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
