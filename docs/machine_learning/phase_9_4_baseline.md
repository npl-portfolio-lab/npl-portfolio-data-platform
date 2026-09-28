```markdown
# Fase 9.4 — Modelo Baseline de Riesgo Crediticio

## 1. Objetivo

El objetivo de esta fase es construir un primer modelo de Machine Learning que funcione como **baseline** o punto de referencia.

Este modelo todavía no pretende ser el modelo final. Su función es permitirnos medir qué tan bien funciona una solución inicial antes de aplicar técnicas de balanceo, ajuste de threshold, optimización de hiperparámetros o modelos más avanzados.

La variable objetivo es `TARGET`:

- `TARGET = 0`: cliente perteneciente a la clase negativa.
- `TARGET = 1`: cliente perteneciente a la clase positiva de riesgo definida por el dataset.

---

## 2. Problema de desbalance

El dataset completo contiene:

| Clase | Clientes | Porcentaje |
|---|---:|---:|
| TARGET = 0 | 282,686 | 91.93% |
| TARGET = 1 | 24,825 | 8.07% |

Existe un fuerte desbalance entre las clases.

Esto significa que **Accuracy por sí sola no es suficiente** para evaluar el modelo.

Por ejemplo, un modelo que predijera siempre:

```text
TARGET = 0
```

obtendría aproximadamente:

```text
Accuracy ≈ 91.93%
```

pero detectaría:

```text
0 clientes TARGET = 1
```

Por esta razón utilizaremos principalmente:

- ROC-AUC
- PR-AUC
- Precision
- Recall
- F1-score
- Matriz de confusión

---

# 3. Datos utilizados

En la Fase 9.2 se realizó un split estratificado:

| Dataset | Clientes |
|---|---:|
| TRAIN | 246,008 |
| VALIDATION | 61,503 |
| TOTAL | 307,511 |

Distribución de `TARGET=1`:

```text
TRAIN:      8.0729%
VALIDATION: 8.0728%
```

Esto confirma que el split conservó aproximadamente la misma proporción de clases.

## Regla utilizada

```text
TRAIN
  ↓
aprender preprocessing
  ↓
entrenar modelo

VALIDATION
  ↓
evaluar modelo
```

El modelo **no aprende de VALIDATION**.

Esto es importante para prevenir **Data Leakage**.

---

# 4. Preprocessing utilizado

Antes del entrenamiento se construyó el pipeline de preprocessing de la Fase 9.3.

Features originales utilizadas:

```text
Features predictoras:   254
Features numéricas:     238
Features categóricas:    16
```

Para las variables numéricas se utilizó:

```text
SimpleImputer(strategy="median")
```

Las medianas se aprendieron exclusivamente utilizando TRAIN.

Para las variables categóricas se utilizó:

```text
SimpleImputer(strategy="most_frequent")
```

seguido de:

```text
OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=True
)
```

Después del One-Hot Encoding se obtuvieron:

```text
378 features
```

---

# 5. Procesamiento sparse y por batches

Los datos transformados se almacenaron como matrices:

```text
CSR sparse matrix
```

Esto evita convertir innecesariamente las features en grandes matrices densas.

El preprocessing fue ejecutado utilizando batches de:

```text
20,000 filas
```

Resultado:

```text
TRAIN
246,008 x 378

VALIDATION
61,503 x 378
```

Posteriormente los batches fueron unidos utilizando:

```text
scipy.sparse.vstack()
```

manteniendo la representación CSR.

---

# 6. Modelo Baseline

El primer modelo utilizado fue:

```text
LogisticRegression
```

Configuración inicial:

```text
threshold = 0.50
class_weight = None
random_state = 42
```

El modelo fue entrenado exclusivamente utilizando:

```text
TRAIN
246,008 clientes
378 features
```

Y evaluado exclusivamente utilizando:

```text
VALIDATION
61,503 clientes
```

---

# 7. Resultados iniciales

El baseline produjo:

| Métrica | Resultado |
|---|---:|
| ROC-AUC | 0.682146 |
| PR-AUC | 0.157146 |
| Precision | 0.032258 |
| Recall | 0.000201 |
| F1-score | 0.000400 |

Estos resultados fueron obtenidos utilizando:

```text
threshold = 0.50
```

---

# 8. Matriz de confusión

La matriz obtenida fue:

| | Predicho 0 | Predicho 1 |
|---|---:|---:|
| Real 0 | 56,508 | 30 |
| Real 1 | 4,964 | 1 |

Por lo tanto:

```text
TN = 56,508
FP =     30
FN =  4,964
TP =      1
```

---

# 9. Interpretación de la matriz

## True Negative — TN

```text
TN = 56,508
```

Son clientes realmente negativos que fueron clasificados correctamente como negativos.

---

## False Positive — FP

```text
FP = 30
```

Son clientes realmente negativos que fueron clasificados incorrectamente como positivos.

---

## False Negative — FN

```text
FN = 4,964
```

Son clientes realmente positivos que el modelo clasificó como negativos.

Este es el principal problema del baseline utilizando threshold `0.50`.

---

## True Positive — TP

```text
TP = 1
```

De los:

```text
4,965 positivos reales
```

el modelo identificó correctamente solamente:

```text
1
```

con threshold `0.50`.

Esto explica el Recall extremadamente bajo.

---

# 10. ¿El modelo no aprendió?

No necesariamente.

Aunque el Recall utilizando threshold `0.50` es prácticamente cero, obtuvimos:

```text
ROC-AUC = 0.682146
```

Esto indica que existe cierta capacidad del modelo para ordenar o separar los clientes positivos y negativos mediante sus scores.

Debemos diferenciar:

```text
CAPACIDAD DE RANKING
        ↓
ROC-AUC / PR-AUC

        VS.

DECISIÓN BINARIA
        ↓
threshold
        ↓
0 o 1
```

Un modelo puede tener cierta capacidad de ranking y al mismo tiempo producir malas clasificaciones binarias si el threshold utilizado no es adecuado para la distribución de sus probabilidades.

---

# 11. Diagnóstico de probabilidades

Para entender el problema se analizaron las probabilidades producidas por el modelo sobre VALIDATION.

Distribución general:

| Percentil | Probabilidad |
|---|---:|
| P0 | 0.000001 |
| P1 | 0.008617 |
| P5 | 0.019533 |
| P25 | 0.043041 |
| P50 | 0.068513 |
| P75 | 0.107424 |
| P90 | 0.154156 |
| P95 | 0.185790 |
| P99 | 0.258508 |
| P100 | 0.999923 |

La mediana de todas las probabilidades está aproximadamente en:

```text
0.0685
```

Esto muestra que la mayoría de scores están muy por debajo de:

```text
0.50
```

---

# 12. Probabilidades según TARGET

Para los clientes:

```text
TARGET = 1
```

se obtuvo:

```text
Clientes: 4,965
Media:    0.115665
Mediana:  0.104383
Máximo:   0.669233
```

Para:

```text
TARGET = 0
```

se obtuvo:

```text
Clientes: 56,538
Media:    0.078864
Mediana:  0.066124
Máximo:   0.999923
```

Comparación:

```text
                  TARGET=1      TARGET=0

Media              0.1157        0.0789
Mediana            0.1044        0.0661
```

Los positivos reciben en promedio scores mayores que los negativos.

Sin embargo, existe solapamiento entre ambas clases.

---

# 13. Problema del threshold 0.50

Con threshold:

```text
0.50
```

solamente:

```text
31 de 61,503 clientes
```

fueron clasificados como positivos.

Eso representa aproximadamente:

```text
0.0504%
```

de VALIDATION.

Por esta razón el modelo prácticamente siempre responde:

```text
TARGET = 0
```

y genera:

```text
4,964 falsos negativos
```

---

# 14. Análisis de diferentes thresholds

Sin reentrenar el modelo se evaluaron diferentes thresholds.

| Threshold | Precision | Recall | F1 | TN | FP | FN | TP |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.0323 | 0.0002 | 0.0004 | 56,508 | 30 | 4,964 | 1 |
| 0.40 | 0.2368 | 0.0036 | 0.0071 | 56,480 | 58 | 4,947 | 18 |
| 0.30 | 0.2715 | 0.0165 | 0.0311 | 56,318 | 220 | 4,883 | 82 |
| 0.20 | 0.2218 | 0.1013 | 0.1391 | 54,773 | 1,765 | 4,462 | 503 |
| 0.15 | 0.1887 | 0.2550 | 0.2169 | 51,094 | 5,444 | 3,699 | 1,266 |
| 0.10 | 0.1478 | 0.5249 | 0.2307 | 41,518 | 15,020 | 2,359 | 2,606 |
| 0.05 | 0.1025 | 0.8594 | 0.1831 | 19,156 | 37,382 | 698 | 4,267 |

---

# 15. ¿Qué ocurre cuando bajamos el threshold?

Cuando disminuimos el threshold:

```text
threshold ↓

más clientes son clasificados como positivos
        ↓
TP aumenta
FN disminuye
        ↓
Recall aumenta
```

Pero también:

```text
threshold ↓
        ↓
FP aumenta
        ↓
Precision puede disminuir
```

Este comportamiento representa el trade-off entre:

```text
Precision ↔ Recall
```

---

# 16. Ejemplo con threshold 0.10

Con:

```text
threshold = 0.10
```

obtenemos:

```text
TP =  2,606
FN =  2,359
FP = 15,020
TN = 41,518
```

Y las métricas:

```text
Precision = 14.78%
Recall    = 52.49%
F1        = 23.07%
```

Ahora el modelo detecta aproximadamente la mitad de los positivos reales.

Sin embargo, aumenta considerablemente la cantidad de falsos positivos.

---

# 17. Ejemplo con threshold 0.05

Con:

```text
threshold = 0.05
```

obtenemos:

```text
TP =  4,267
FN =    698
FP = 37,382
TN = 19,156
```

Recall:

```text
85.94%
```

El modelo detecta una gran proporción de positivos.

Pero genera:

```text
37,382 falsos positivos
```

Por lo tanto, bajar el threshold no es gratis.

---

# 18. ¿Cuál threshold debemos utilizar?

Todavía **no se selecciona un threshold definitivo**.

Dentro de los thresholds evaluados, `0.10` produjo el mayor F1:

```text
F1 = 0.2307
```

Pero esto solamente significa que fue el mayor F1 entre los thresholds probados.

No significa automáticamente que sea el threshold correcto para el problema.

La decisión debe considerar el costo de negocio de:

```text
False Positive
vs.
False Negative
```

---

# 19. Interpretación en riesgo crediticio

Si `TARGET=1` representa el evento de riesgo que queremos detectar:

## False Negative

```text
Real:      TARGET = 1
Predicción TARGET = 0
```

El modelo no detectó un cliente que realmente pertenecía a la clase de riesgo.

## False Positive

```text
Real:      TARGET = 0
Predicción TARGET = 1
```

El modelo clasificó como riesgoso a un cliente que realmente pertenecía a la clase negativa.

Por esta razón la elección del threshold debe estar relacionada con el objetivo y costo del negocio.

---

# 20. ROC-AUC

Resultado:

```text
ROC-AUC = 0.682146
```

ROC-AUC mide la capacidad del modelo para discriminar o rankear las clases considerando diferentes thresholds.

No evalúa únicamente la decisión producida con threshold `0.50`.

Por eso podemos tener simultáneamente:

```text
ROC-AUC = 0.6821
```

y:

```text
Recall @ 0.50 ≈ 0
```

No existe contradicción.

---

# 21. PR-AUC

Resultado:

```text
PR-AUC = 0.157146
```

PR-AUC resume el comportamiento de:

```text
Precision
+
Recall
```

a través de diferentes thresholds.

Esta métrica es especialmente importante en nuestro problema porque la clase positiva representa solamente aproximadamente:

```text
8.07%
```

del dataset.

Por lo tanto, PR-AUC será una de las métricas importantes para comparar los siguientes modelos.

---

# 22. Conclusión actual

Hasta este punto el pipeline funciona de la siguiente manera:

```text
FEATURE ENGINEERING
        ↓
254 features predictoras
        ↓
SPLIT ESTRATIFICADO
        ↓
TRAIN / VALIDATION
        ↓
PREPROCESSING
aprendido SOLO con TRAIN
        ↓
One-Hot Encoding
        ↓
378 features
        ↓
Matrices CSR
        ↓
LOGISTIC REGRESSION
        ↓
entrenamiento con TRAIN
        ↓
VALIDATION
        ↓
probabilidades
        ↓
ROC-AUC / PR-AUC
        ↓
threshold
        ↓
Precision / Recall / F1
        ↓
Matriz de confusión
```

El baseline obtuvo:

```text
ROC-AUC = 0.682146
PR-AUC  = 0.157146
```

Esto demuestra que existe señal predictiva.

Sin embargo, utilizar:

```text
threshold = 0.50
```

produce un Recall prácticamente nulo debido a la distribución de scores y al desbalance existente.

El análisis de thresholds confirmó el trade-off entre Precision y Recall.

---

# 23. Qué NO debemos concluir todavía

Todavía no podemos afirmar que:

```text
threshold = 0.10 es el threshold final
```

Tampoco podemos afirmar que:

```text
Logistic Regression será el modelo final
```

El propósito de esta fase es establecer una referencia medible.

A partir de ahora los experimentos deben compararse contra este baseline.

---

# 24. Próximo experimento

El siguiente experimento mantendrá constantes:

```text
mismo TRAIN
mismo VALIDATION
mismas 378 features
mismo preprocessing
mismo algoritmo
```

La única modificación será el tratamiento del desbalance:

```python
class_weight="balanced"
```

Compararemos:

```text
BASELINE A
LogisticRegression
class_weight=None

        VS.

BASELINE B
LogisticRegression
class_weight="balanced"
```

Evaluando:

```text
ROC-AUC
PR-AUC
Precision
Recall
F1-score
Matriz de confusión
```

Esto permitirá determinar qué efecto tiene el balanceo de clases sin mezclar simultáneamente otros cambios.

---

# 25. Principio de experimentación

A partir de esta fase seguiremos una regla:

```text
Cambiar una cosa
        ↓
medir
        ↓
comparar
        ↓
entender el resultado
        ↓
documentar
        ↓
continuar
```

Esto permite construir el modelo de forma reproducible y entender por qué cada decisión mejora o empeora los resultados.
```

