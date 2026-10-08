# Fase 10.9 — Bootstrap pareado V2

## 1. Objetivo

Estimar la incertidumbre de las diferencias de rendimiento
entre los modelos logísticos balanceado y sin balanceo.

## 2. Datos y metodología

- Partición: VALIDATION V2.
- Registros: 49,202.
- Positivos: 3,972.
- Variables: 378.
- Método: paired_stratified_bootstrap.
- Intervalo: percentile.
- Réplicas: 1,000.
- Confianza nominal: 95%.
- Semilla: 42.
- TEST: no utilizado.

La diferencia se define como:

Delta = métrica balanceado - métrica sin balanceo.

El remuestreo es pareado porque ambos modelos se
evalúan sobre los mismos índices en cada réplica.

Se conserva la cantidad original de cada clase.

## 3. Resultados

| Métrica | Delta observado | IC 95 % | Incluye cero |
|---|---:|---|---|
| roc_auc | +0.000412 | [-0.000253, +0.001071] | Sí |
| pr_auc | -0.001273 | [-0.002373, -0.000032] | No |
| precision_at_k_1000 | -0.009000 | [-0.018000, +0.006000] | Sí |
| precision_at_k_3000 | +0.000667 | [-0.004333, +0.006000] | Sí |
| precision_at_k_5000 | -0.000200 | [-0.003800, +0.003205] | Sí |

## 4. Interpretación

Los intervalos de ROC-AUC y Precision@K incluyen cero.

El intervalo de PR-AUC queda ligeramente por debajo
de cero, favoreciendo al modelo sin balanceo.

Este resultado debe interpretarse como exploratorio.

No se aplicaron correcciones por comparaciones
múltiples. No se declara superioridad definitiva.

## 5. Limitaciones

- Se utilizaron intervalos percentiles bootstrap.
- No se corrigieron las cinco comparaciones múltiples.
- No se realizó validación temporal independiente.
- TEST V2 permanece reservado.
- TEST V2 no es completamente inédito respecto
  a los experimentos históricos.
- La auditoría temporal de variables sigue pendiente.
- TARGET representa dificultad de pago y no
  recuperación efectiva de cartera castigada.
- El bootstrap cuantifica incertidumbre condicional
  al conjunto VALIDATION y no elimina posibles
  sesgos del diseño de datos.

## 6. Reproducibilidad

Código:

- `src/npl_portfolio/ml/paired_bootstrap.py`
- `scripts/ml/compare_models_bootstrap_v2.py`
- `scripts/ml/audit_bootstrap_v2.py`
- `tests/ml/test_paired_bootstrap.py`

Resultados:

- `data/artifacts/ml/v2/bootstrap_comparison.json`

## 7. Conclusión

No se establece un modelo ganador definitivo.

Las diferencias Top-K son pequeñas y sus intervalos
incluyen cero para las tres capacidades analizadas.

La selección operativa debe considerar también
capacidad, costos, estabilidad y calibración.

Este documento se genera automáticamente desde
el JSON de resultados.
