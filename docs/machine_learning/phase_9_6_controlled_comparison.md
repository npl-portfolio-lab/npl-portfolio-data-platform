# Fase 9.6 — Comparación controlada de modelos

**Proyecto:** NPL Portfolio Data Platform  
**Fecha de generación:** 2026-10-08 17:45 UTC  
**Estado:** Experimento completado

## 1. Objetivo

Evaluar el efecto del balanceo de clases en una regresión logística
que predice dificultades de pago (`TARGET = 1`).

La comparación utiliza el mismo conjunto de entrenamiento,
el mismo conjunto de validación y la misma configuración técnica.

La única diferencia experimental es el parámetro `class_weight`.

## 2. Configuración del experimento

| Parámetro | Valor |
|---|---|
| Algoritmo | LogisticRegression |
| Escalador | MaxAbsScaler |
| Solver | liblinear |
| max_iter | 200 |
| tol | 0.001 |
| random_state | 42 |
| Threshold | 0.5 |
| Modelo A | class_weight=None |
| Modelo B | class_weight=balanced |

El escalador se ajusta exclusivamente con TRAIN y posteriormente
se aplica a VALIDATION para evitar fuga de información.

## 3. Resultados de evaluación

| Métrica | Sin balanceo | Balanceado |
|---|---:|---:|
| ROC-AUC | 77.22 % | 77.18 % |
| PR-AUC | 26.33 % | 25.97 % |
| Precision | 57.62 % | 17.41 % |
| Recall | 3.12 % | 69.83 % |
| F1-score | 5.92 % | 27.87 % |

## 4. Matriz de confusión

| Resultado | Sin balanceo | Balanceado |
|---|---:|---:|
| Verdaderos negativos (TN) | 56,424 | 40,094 |
| Falsos positivos (FP) | 114 | 16,444 |
| Falsos negativos (FN) | 4,810 | 1,498 |
| Verdaderos positivos (TP) | 155 | 3,467 |

## 5. Interpretación

### Modelo sin balanceo

- Precision: 57.62 %.
- Recall: 3.12 %.
- Identifica correctamente 155 casos positivos.
- No identifica 4,810 casos positivos.
- Genera 114 falsos positivos.

Este modelo es conservador al clasificar clientes como positivos.
Aunque sus predicciones positivas son relativamente precisas,
detecta una proporción muy pequeña de los casos reales.

### Modelo balanceado

- Precision: 17.41 %.
- Recall: 69.83 %.
- Identifica correctamente 3,467 casos positivos.
- No identifica 1,498 casos positivos.
- Genera 16,444 falsos positivos.

Este modelo detecta una proporción considerablemente mayor
de clientes con dificultades de pago, a costa de incrementar
las alertas incorrectas.

## 6. Diferencias principales

Al utilizar balanceo de clases:

- Se detectan 3,312 casos positivos adicionales.
- Se reducen los falsos negativos en 3,312.
- Se generan 16,330 falsos positivos adicionales.
- El Recall aumenta de 3.12 %
  a 69.83 %.
- El F1-score aumenta de 5.92 %
  a 27.87 %.

Las diferencias en ROC-AUC y PR-AUC son pequeñas.
El cambio más importante aparece en las decisiones
de clasificación al utilizar el umbral 0.50.

## 7. Conclusiones

El balanceo de clases mejora significativamente la capacidad
de detección de la clase positiva al umbral evaluado.

Sin embargo, esta mejora implica un incremento considerable
de falsos positivos, que podrían traducirse en revisiones
innecesarias o decisiones de riesgo incorrectas.

Por tanto, no se selecciona todavía un modelo definitivo
para su utilización operativa.

Ambos modelos deben considerarse en la siguiente fase,
donde se evaluarán diferentes umbrales de decisión,
el comportamiento de los errores y sus posibles
implicaciones para el negocio.

## 8. Próxima fase

**Fase 10 — Evaluación avanzada y análisis de umbrales.**

Se analizará el compromiso entre Precision y Recall,
la cantidad de falsos positivos y falsos negativos
y los criterios para seleccionar un umbral adecuado.

---

*Documento generado automáticamente a partir de los resultados
del experimento de comparación controlada.*
