# Fase 9.6 — Regresión Logística balanceada

**Informe generado:** 2026-10-08 17:29 UTC

## 1. Objetivo

Evaluar el desempeño de un modelo de clasificación para identificar clientes con dificultades de pago (`TARGET = 1`).

## 2. Configuración del experimento

| Parámetro | Valor |
|---|---|
| Modelo | logistic_regression_balanced |
| Dataset evaluado | validation |
| Registros evaluados | 61,503 |
| Variables | 378 |
| Escalador | MaxAbsScaler |
| Solver | liblinear |
| Balanceo de clases | balanced |
| Iteraciones | 9 |
| Umbral de clasificación | 0.5 |

## 3. Resultados del entrenamiento

El entrenamiento registró una duración de **36.45 segundos**.

**Advertencia de convergencia:** No

## 4. Métricas de evaluación

| Métrica | Resultado | ¿Qué mide? |
|---|---:|---|
| ROC-AUC | 77.18% | Capacidad de distinguir entre clases |
| PR-AUC | 25.97% | Precisión y cobertura de la clase positiva |
| Precision | 17.41% | Proporción de predicciones positivas correctas |
| Recall | 69.83% | Proporción de positivos reales detectados |
| F1-score | 27.87% | Equilibrio entre precision y recall |

## 5. Matriz de confusión

| Resultado | Cantidad | Interpretación |
|---|---:|---|
| TN | 40,094 | Negativos correctamente identificados |
| FP | 16,444 | Negativos clasificados como positivos |
| FN | 1,498 | Positivos que el modelo no detectó |
| TP | 3,467 | Positivos correctamente identificados |

## 6. Interpretación de los resultados

El modelo identificó correctamente **3,467** de **4,965** casos positivos reales, equivalentes a un recall de **69.83%**.

De los **19,911** casos clasificados como positivos, **3,467** fueron correctos. Esto representa una precision de **17.41%**.

Se registraron **16,444 falsos positivos** y **1,498 falsos negativos**.

## 7. Conclusiones y limitaciones

El modelo permite identificar una proporción de los casos positivos, pero también genera errores de clasificación que deben evaluarse según su impacto en el negocio.

Estas métricas corresponden al conjunto de validación. No constituyen una evaluación sobre datos futuros ni garantizan el mismo desempeño en producción.

La selección del modelo requiere comparar alternativas bajo condiciones equivalentes y analizar el umbral de clasificación.
