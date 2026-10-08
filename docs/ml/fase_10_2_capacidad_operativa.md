# Fase 10.2 — Análisis de capacidad operativa

## Objetivo

Seleccionar, entre las configuraciones evaluadas, el modelo y threshold que detecten más positivos reales sin superar un límite de alertas.

## Datos

- Dataset: `validation`
- Registros: 61,503
- Positivos reales: 4,965

## Metodología

Para cada capacidad máxima se seleccionó la configuración con más verdaderos positivos, siempre que las alertas no superaran el límite. En caso de empate se priorizó la configuración con menos falsos positivos.

## Resultados

| Capacidad | Modelo | Threshold | Alertas | TP | FP | Precision | Recall | Uso de capacidad |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | unbalanced | 0.40 | 769 | 362 | 407 | 0.4707 | 0.0729 | 76.90% |
| 3,000 | unbalanced | 0.30 | 1,988 | 778 | 1,210 | 0.3913 | 0.1567 | 66.27% |
| 5,000 | balanced | 0.75 | 4,934 | 1,510 | 3,424 | 0.3060 | 0.3041 | 98.68% |

## Interpretación

El modelo seleccionado puede cambiar según la capacidad disponible. Una mayor capacidad permite revisar más registros, pero también puede incrementar los falsos positivos.

## Limitaciones

- Las capacidades representan alertas sobre el dataset completo de validación, no volúmenes diarios.
- La selección utiliza únicamente los thresholds evaluados en la Fase 10.1.
- Puede quedar capacidad sin utilizar.
- Los resultados no incorporan costos de revisión ni pérdidas económicas por errores.
- No representan una validación independiente en producción.

## Próxima etapa

Fase 10.3: evaluación Top-K para comparar modelos utilizando exactamente la misma cantidad de alertas.

## Artefacto

`data/artifacts/ml/operational_capacity_analysis.json`
