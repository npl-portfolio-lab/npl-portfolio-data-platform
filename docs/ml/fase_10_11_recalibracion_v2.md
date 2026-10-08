# Fase 10.11 — Recalibración de probabilidades V2

## Objetivo

Comparar probabilidades originales, Sigmoid e Isotonic
para los dos modelos de regresión logística V2.

## Protocolo experimental

- Fuente: VALIDATION V2.
- Separación: estratificada, 50 % FIT y 50 % EVAL.
- Semilla: 42.
- FIT: 24,601 clientes.
- EVAL: 24,601 clientes.
- Positivos FIT: 1,986.
- Positivos EVAL: 1,986.
- Clientes compartidos: 0.
- TEST V2: no utilizado.

Los modelos originales permanecen sin modificaciones.

Los calibradores se ajustan exclusivamente sobre FIT
y se evalúan exclusivamente sobre EVAL.

## Resultados

| Modelo | Método | Brier ↓ | Log Loss ↓ | ROC-AUC ↑ | PR-AUC ↑ |
|---|---|---:|---:|---:|---:|
| unbalanced | original | 0.067481 | 0.243415 | 0.769240 | 0.247641 |
| unbalanced | sigmoid | 0.068546 | 0.249971 | 0.769240 | 0.247641 |
| unbalanced | isotonic | 0.067593 | 0.243872 | 0.767813 | 0.228366 |
| balanced | original | 0.195268 | 0.575348 | 0.769290 | 0.246715 |
| balanced | sigmoid | 0.067493 | 0.243553 | 0.769290 | 0.246715 |
| balanced | isotonic | 0.067608 | 0.243942 | 0.767978 | 0.230144 |

## Interpretación

El menor Brier Score observado corresponde a:

- Modelo: unbalanced.
- Método: original.
- Brier Score: 0.067481.

La calibración Sigmoid mejora notablemente las
probabilidades del modelo balanceado.

En el modelo sin balanceo, los métodos adicionales
no mejoran el Brier Score respecto al original.

Isotonic presenta una disminución de PR-AUC
en ambos modelos.

## Decisión provisional

Mantener como candidato principal el modelo
sin balanceo con probabilidades originales.

Conservar el modelo balanceado calibrado mediante
Sigmoid como alternativa experimental.

No se declara superioridad estadística definitiva.

## Limitaciones

- VALIDATION V2 ya se utilizó en análisis anteriores.
- La evaluación es independiente del ajuste de
  los calibradores, pero no de toda selección previa.
- No se realizó comparación estadística pareada
  de las mejoras de calibración.
- La calibración se realizó sobre las probabilidades
  originales, no directamente sobre las puntuaciones
  de decisión del modelo.
- TEST V2 permanece reservado.
- El conjunto TEST V2 no es completamente inédito
  respecto a experimentos históricos.
- La auditoría de posibles fugas temporales
  en las variables sigue pendiente.
- TARGET mide dificultad de pago, no recuperación
  real de cartera castigada.

## Artefactos

- `data/artifacts/ml/v2/calibration/calibrators_comparison.json`
- `data/artifacts/ml/v2/calibration/unbalanced_sigmoid_calibrator.joblib`
- `data/artifacts/ml/v2/calibration/unbalanced_isotonic_calibrator.joblib`
- `data/artifacts/ml/v2/calibration/balanced_sigmoid_calibrator.joblib`
- `data/artifacts/ml/v2/calibration/balanced_isotonic_calibrator.joblib`

## Código

- `src/npl_portfolio/ml/probability_calibrator.py`
- `scripts/ml/train_calibrators_v2.py`
- `scripts/ml/audit_calibrators_v2.py`
- `tests/ml/test_probability_calibrator.py`

Documento generado automáticamente desde el reporte JSON.
