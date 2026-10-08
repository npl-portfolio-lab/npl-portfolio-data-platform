# Fase 10.10 — Evaluación de calibración V2

## 1. Objetivo

Evaluar la calidad de las probabilidades originales
de los modelos logísticos V2.

## 2. Datos

- Partición: VALIDATION V2.
- Registros: 49,202.
- Casos positivos: 3,972.
- Variables: 378.
- Intervalos de calibración: 10.
- TEST: no utilizado.

## 3. Metodología

Se evalúan las probabilidades originales, sin aplicar
un calibrador adicional.

Brier Score: error cuadrático medio probabilístico.

Log Loss: pérdida logarítmica probabilística.

Ambas métricas se minimizan.

## 4. Resultados

| Modelo | Brier Score | Log Loss | Probabilidad media | Prevalencia |
|---|---:|---:|---:|---:|
| unbalanced | 0.067636 | 0.244028 | 8.1572% | 8.0728% |
| balanced | 0.195256 | 0.575490 | 40.6413% | 8.0728% |

## 5. Análisis por intervalos

### Modelo unbalanced

| Intervalo | Clientes | Probabilidad media | Frecuencia observada |
|---|---:|---:|---:|
| [0.0, 0.1] | 36,491 | 4.1860% | 4.1490% |
| [0.1, 0.2] | 8,333 | 13.9078% | 14.4726% |
| [0.2, 0.3] | 2,705 | 24.2800% | 23.7338% |
| [0.3, 0.4] | 1,014 | 34.3825% | 33.0375% |
| [0.4, 0.5] | 435 | 44.2083% | 39.3103% |
| [0.5, 0.6] | 160 | 54.1360% | 41.2500% |
| [0.6, 0.7] | 48 | 64.1056% | 60.4167% |
| [0.7, 0.8] | 15 | 73.1238% | 53.3333% |
| [0.9, 1.0] | 1 | 97.5630% | 100.0000% |
### Modelo balanced

| Intervalo | Clientes | Probabilidad media | Frecuencia observada |
|---|---:|---:|---:|
| [0.0, 0.1] | 2,325 | 7.1727% | 0.7742% |
| [0.1, 0.2] | 7,591 | 15.2803% | 1.6072% |
| [0.2, 0.3] | 8,621 | 24.9301% | 2.8999% |
| [0.3, 0.4] | 7,736 | 34.8713% | 4.4984% |
| [0.4, 0.5] | 6,923 | 44.8805% | 6.8179% |
| [0.5, 0.6] | 5,494 | 54.8439% | 9.6469% |
| [0.6, 0.7] | 4,532 | 64.7646% | 15.0927% |
| [0.7, 0.8] | 3,311 | 74.7031% | 20.5980% |
| [0.8, 0.9] | 2,150 | 84.3719% | 30.0000% |
| [0.9, 1.0] | 519 | 92.4909% | 42.5819% |

Los intervalos son de ancho uniforme y se omiten
aquellos sin registros.

## 6. Interpretación

El modelo sin balanceo presenta menores valores de
Brier Score y Log Loss en VALIDATION V2.

El modelo balanceado presenta una probabilidad
media considerablemente superior a la prevalencia
observada, indicando sobreestimación agregada.

La proximidad entre probabilidad media y prevalencia
no garantiza calibración adecuada en todos los
intervalos.

## 7. Limitaciones

- Evaluación realizada sobre VALIDATION V2.
- No se ha ajustado ningún calibrador adicional.
- Los resultados son descriptivos.
- La selección previa de modelos utilizó VALIDATION.
- TEST V2 permanece reservado.
- La auditoría temporal de variables sigue pendiente.
- TARGET representa dificultad de pago, no
  recuperación efectiva de cartera castigada.

## 8. Próximo paso

Estudiar métodos de calibración usando datos
separados para ajuste y evaluación.

Cualquier comparación de calibradores debe evitar
evaluarlos sobre los registros empleados para
ajustarlos.

## 9. Reproducibilidad

- `src/npl_portfolio/ml/calibration_evaluator.py`
- `scripts/ml/evaluate_calibration_v2.py`
- `scripts/ml/audit_calibration_v2.py`
- `tests/ml/test_calibration_evaluator.py`
- `data/artifacts/ml/v2/calibration_evaluation.json`

Documento generado automáticamente.
