# Fase 9.5 — Evaluación del modelo baseline

**Informe generado:** 2026-10-08 17:32 UTC

## 1. Objetivo

Evaluar el desempeño de un modelo de clasificación para identificar clientes con dificultades de pago (`TARGET = 1`).

## 2. Configuración del experimento

| Parámetro | Valor |
|---|---|
| Modelo | logistic_regression_baseline |
| Dataset evaluado | validation |
| Registros evaluados | 61,503 |
| Variables | 378 |
| Escalador | No registrado |
| Solver | No registrado |
| Balanceo de clases | No registrado |
| Iteraciones | No registrado |
| Umbral de clasificación | 0.5 |

## 3. Resultados del entrenamiento

El entrenamiento registró una duración de **0.00 segundos**.

**Advertencia de convergencia:** No

## 4. Métricas de evaluación

| Métrica | Resultado | ¿Qué mide? |
|---|---:|---|
| ROC-AUC | 68.21% | Capacidad de distinguir entre clases |
| PR-AUC | 15.71% | Precisión y cobertura de la clase positiva |
| Precision | 3.23% | Proporción de predicciones positivas correctas |
| Recall | 0.02% | Proporción de positivos reales detectados |
| F1-score | 0.04% | Equilibrio entre precision y recall |

## 5. Matriz de confusión

| Resultado | Cantidad | Interpretación |
|---|---:|---|
| TN | 56,508 | Negativos correctamente identificados |
| FP | 30 | Negativos clasificados como positivos |
| FN | 4,964 | Positivos que el modelo no detectó |
| TP | 1 | Positivos correctamente identificados |

## 6. Interpretación de los resultados

El modelo identificó correctamente **1** de **4,965** casos positivos reales, equivalentes a un recall de **0.02%**.

De los **31** casos clasificados como positivos, **1** fueron correctos. Esto representa una precision de **3.23%**.

Se registraron **30 falsos positivos** y **4,964 falsos negativos**.

## 7. Conclusiones y limitaciones

El modelo permite identificar una proporción de los casos positivos, pero también genera errores de clasificación que deben evaluarse según su impacto en el negocio.

Estas métricas corresponden al conjunto de validación. No constituyen una evaluación sobre datos futuros ni garantizan el mismo desempeño en producción.

La selección del modelo requiere comparar alternativas bajo condiciones equivalentes y analizar el umbral de clasificación.
