# Fase 10.4 — Estabilidad estadística mediante bootstrap pareado

## Objetivo

Estimar la incertidumbre de las diferencias entre los modelos sin balanceo y balanceado en la priorización Top-K.

## Configuración

- Dataset: `validation`
- Registros: 61,503
- Repeticiones: 1,000
- Semilla aleatoria: 42
- Nivel de confianza: 95%
- Método: `paired_nonparametric_bootstrap_percentile`
- Diferencia: balanceado menos sin balanceo.

## Metodología

En cada repetición se generó una muestra con reemplazo del conjunto VALIDATION. Ambos modelos se evaluaron sobre los mismos índices, recalculando el ranking Top-K.

Los intervalos se calcularon mediante percentiles 2.5 % y 97.5 % de las diferencias bootstrap.

## Resultados

| K | Diferencia media Precision@K | IC 95 % inferior | IC 95 % superior | Excluye cero |
|---:|---:|---:|---:|---|
| 1,000 | 0.001087 | -0.013000 | 0.013000 | No |
| 3,000 | -0.003162 | -0.009000 | 0.002675 | No |
| 5,000 | -0.000725 | -0.004200 | 0.003000 | No |

## Interpretación

Si el intervalo de confianza de la diferencia incluye el cero, los resultados no aportan evidencia suficiente para afirmar una diferencia consistente bajo este método.

Un intervalo que excluya cero sugiere una diferencia sistemática en las remuestras, aunque no garantiza superioridad futura ni utilidad económica.

## Limitaciones

- Bootstrap estima incertidumbre condicionada a VALIDATION.
- No sustituye una evaluación en TEST independiente.
- Los modelos se mantienen fijos; no se reentrenan.
- Se asume que las observaciones pueden remuestrearse de manera independiente.
- Los intervalos percentiles pueden tener limitaciones en muestras sesgadas o con dependencia temporal.
- Los empates de puntuación se resuelven mediante orden estable.
- No se incorporan costos financieros.

## Artefactos

- `data/artifacts/ml/bootstrap_stability_evaluation.json`
- `scripts/ml/evaluate_bootstrap_stability.py`
