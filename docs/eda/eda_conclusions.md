# Conclusiones del Análisis Exploratorio de Datos

## 1. Objetivo

El Análisis Exploratorio de Datos (EDA) tuvo como propósito comprender la
estructura, calidad, cobertura y comportamiento de las variables disponibles
en el dataset Home Credit Default Risk antes de iniciar la etapa de Feature
Engineering y el desarrollo de modelos de Machine Learning.

El análisis se realizó sobre la tabla principal `application_train` y las
fuentes históricas:

- `bureau`
- `bureau_balance`
- `previous_application`
- `POS_CASH_balance`
- `credit_card_balance`
- `installments_payments`

Las conclusiones obtenidas en esta fase determinan las decisiones que deberán
aplicarse durante Feature Engineering y modelado.

---

## 2. Interpretación de la variable objetivo

La variable `TARGET` de Home Credit debe interpretarse como una variable de
riesgo crediticio asociada a dificultades de pago del cliente.

En `application_train` se observaron:

| TARGET | Clientes | Porcentaje |
|---|---:|---:|
| 0 | 282,686 | 91.93 % |
| 1 | 24,825 | 8.07 % |

Existe, por tanto, un desbalance considerable entre las clases.

Este desbalance implica que durante la evaluación de modelos no será adecuado
utilizar únicamente `accuracy`. Será necesario considerar métricas como:

- ROC-AUC
- Precision
- Recall
- F1-score
- PR-AUC

### Limitación respecto al objetivo NPL

`TARGET` no representa directamente:

- probabilidad de recuperación;
- Loss Given Default (LGD);
- monto recuperado;
- flujo de recuperación;
- tiempo hasta recuperación.

Por esta razón, los modelos entrenados directamente con Home Credit no deberán
presentarse como modelos de recuperación de cartera NPL.

Home Credit permitirá desarrollar el componente de riesgo crediticio del
proyecto. La posterior valoración de cartera y simulación financiera deberá
incorporar supuestos de recuperación explícitos o una fuente de datos adicional
que contenga resultados reales de recuperación.

---

## 3. Calidad y completitud de `application_train`

`application_train` contiene:

- 307,511 clientes;
- 122 columnas.

Se identificaron:

- 67 columnas con valores faltantes;
- 50 columnas con al menos 30 % de valores faltantes;
- 41 columnas con al menos 50 % de valores faltantes.

Varias variables relacionadas con características de vivienda presentan niveles
de ausencia cercanos o superiores al 60 %.

La existencia de valores faltantes no implica automáticamente que una variable
deba eliminarse. Durante Feature Engineering deberá evaluarse:

1. capacidad predictiva;
2. porcentaje de ausencia;
3. significado de la ausencia;
4. estrategia de imputación;
5. posible creación de indicadores de ausencia.

---

## 4. Valores especiales y distribuciones extremas

Se identificaron distribuciones altamente asimétricas en variables financieras,
incluyendo ingresos, crédito y anualidad.

Por ejemplo, `AMT_INCOME_TOTAL` presenta valores extremos muy superiores a su
mediana.

Estos valores no fueron eliminados durante el EDA.

Para las visualizaciones se utilizaron límites de representación cuando fue
necesario, pero esto no modificó los datos originales.

Durante Feature Engineering deberán evaluarse estrategias como:

- transformaciones logarítmicas;
- winsorización controlada;
- clipping;
- variables derivadas;
- modelos robustos a outliers.

La decisión deberá realizarse sin modificar arbitrariamente información válida.

---

## 5. Valor sentinel en `DAYS_EMPLOYED`

Se identificó el valor:

`DAYS_EMPLOYED = 365243`

en aproximadamente 18 % de los clientes.

Este valor no representa una duración laboral real y deberá tratarse como un
valor especial.

Durante Feature Engineering se recomienda:

1. crear un indicador binario que identifique el sentinel;
2. reemplazar el valor sentinel por `NaN` para los cálculos estadísticos;
3. conservar el indicador para no perder la información implícita asociada a
   su presencia.

Ejemplo conceptual:

`DAYS_EMPLOYED_ANOMALY`

---

## 6. Cobertura de las fuentes históricas

Las fuentes históricas presentan niveles de cobertura diferentes respecto a
los clientes de `application_train`.

Aproximadamente:

| Fuente | Cobertura |
|---|---:|
| Previous Application | 94.65 % |
| Installments | 94.84 % |
| POS CASH | 94.12 % |
| Bureau | 85.69 % |
| Bureau Balance | 29.99 % |
| Credit Card | 28.26 % |

Las diferencias de cobertura son relevantes porque la ausencia de historial
también puede contener información.

Por esta razón, durante Feature Engineering deberán crearse indicadores de
disponibilidad de historial, por ejemplo:

- `HAS_BUREAU_HISTORY`
- `HAS_PREVIOUS_HISTORY`
- `HAS_POS_HISTORY`
- `HAS_CREDIT_CARD_HISTORY`
- `HAS_INSTALLMENTS_HISTORY`
- `HAS_BUREAU_BALANCE_HISTORY`

Esto permitirá distinguir entre un valor agregado realmente igual a cero y la
ausencia completa de registros históricos.

---

## 7. Bureau

`bureau` contiene información sobre créditos reportados previamente para los
clientes.

Se identificaron aproximadamente:

- 1.72 millones de registros;
- 305,811 clientes;
- 5.61 registros por cliente en promedio.

Los clientes con `TARGET = 1` presentan asociaciones con:

- mayor presencia relativa de créditos activos;
- menor presencia de créditos cerrados;
- mayor deuda histórica;
- mayor presencia de eventos de mora;
- historial crediticio relativamente más reciente.

Estos resultados sugieren que las agregaciones de Bureau pueden aportar señales
relevantes al modelo.

Las variables candidatas incluyen conteos de créditos, deuda, crédito total,
mora y antigüedad del historial.

Estas asociaciones son descriptivas y no deben interpretarse como relaciones
causales.

---

## 8. Previous Application

`previous_application` contiene aproximadamente:

- 1.67 millones de solicitudes anteriores;
- 338,857 clientes.

La distribución general observada fue aproximadamente:

- Approved: 62.07 %
- Canceled: 18.94 %
- Refused: 17.40 %
- Unused offer: 1.58 %

Los clientes con `TARGET = 1` presentan:

- menor tasa media de aprobación;
- mayor tasa media de rechazo;
- diferencias en montos y características de solicitudes anteriores;
- solicitudes históricas relativamente más recientes.

Las tasas de aprobación, rechazo y cancelación son candidatas naturales para
Feature Engineering.

---

## 9. POS CASH

`POS_CASH_balance` contiene aproximadamente:

- 10 millones de registros;
- 337,252 clientes;
- 29.66 registros por cliente en promedio.

La mayoría de registros corresponden a contratos activos.

Las variables de días en mora presentan distribuciones concentradas en cero,
pero con colas extremas.

Los clientes con `TARGET = 1` muestran mayores valores medios en indicadores
como:

- `POS_DPD_RATE`;
- `POS_DPD_DEF_RATE`.

Esto indica que la frecuencia de eventos de mora histórica constituye una señal
potencialmente útil para el modelado de riesgo.

---

## 10. Credit Card

`credit_card_balance` contiene aproximadamente:

- 3.84 millones de registros;
- 103,558 clientes;
- 37.08 registros por cliente en promedio.

La cobertura respecto a `application_train` es cercana al 28 %, por lo que la
existencia o ausencia de historial de tarjeta debe conservarse explícitamente.

Los clientes con `TARGET = 1` presentan asociaciones con:

- mayor saldo medio;
- mayor utilización de crédito;
- mayor utilización máxima;
- diferencias en comportamiento de pagos y mora.

Las variables de utilización resultan especialmente relevantes como candidatas
para Feature Engineering.

Debido a la baja cobertura, estas variables deberán combinarse con un indicador
`HAS_CREDIT_CARD_HISTORY`.

---

## 11. Installments Payments

`installments_payments` contiene aproximadamente:

- 13.61 millones de registros;
- 339,587 clientes;
- 40.06 registros por cliente en promedio.

Se derivaron dos conceptos importantes:

### Retraso de pago

`PAYMENT_DELAY_DAYS = DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT`

Interpretación:

- mayor que 0: pago tardío;
- igual a 0: pago en la fecha programada;
- menor que 0: pago anticipado.

### Diferencia de pago

`PAYMENT_DIFFERENCE = AMT_INSTALMENT - AMT_PAYMENT`

Interpretación:

- mayor que 0: pago insuficiente;
- igual a 0: pago completo;
- menor que 0: pago superior al valor programado.

Los clientes con `TARGET = 1` presentan:

- mayor tasa de pagos tardíos;
- mayor tasa de pagos insuficientes;
- menor `PAYMENT_RATIO`;
- menor proporción de sobrepagos;
- menor cantidad promedio de registros y contratos históricos.

Esta fuente presenta alta cobertura y señales de comportamiento directamente
relacionadas con el cumplimiento de pagos, por lo que será una de las fuentes
más relevantes para Feature Engineering.

---

## 12. Bureau Balance

`bureau_balance` contiene aproximadamente:

- 27.30 millones de registros;
- 817,395 identificadores `SK_ID_BUREAU`.

La tabla no contiene directamente `SK_ID_CURR`, por lo que su información debe
relacionarse con el cliente mediante `bureau`.

Se observó que aproximadamente:

- 88.57 % de los registros pueden mapearse a `bureau`;
- 11.43 % no tienen correspondencia disponible;
- existen aproximadamente 43,041 identificadores `SK_ID_BUREAU` sin
  correspondencia.

Los registros sin correspondencia deben conservarse en la fuente procesada,
pero no pueden agregarse de forma confiable a un cliente sin información
adicional.

Entre los clientes que sí pueden analizarse, `TARGET = 1` presenta mayores
valores medios en:

- `BB_DPD_RATE`;
- `BB_SEVERE_DPD_RATE`;
- cantidad de registros con mora;
- nivel máximo de mora.

También se observaron diferencias en la tasa de estados cerrados.

La cobertura final de clientes es cercana al 30 %, por lo que deberá utilizarse
`HAS_BUREAU_BALANCE_HISTORY`.

La interpretación financiera específica de los códigos `STATUS` deberá
mantenerse alineada con la documentación oficial del dataset antes de utilizar
dichos códigos para reglas financieras o de negocio.

---

## 13. Integridad referencial

Durante la fase de calidad de datos se identificaron brechas de cobertura entre
algunas tablas históricas.

Se observaron registros cuyo `SK_ID_PREV` no aparece en
`previous_application`:

- POS CASH: aproximadamente 3.41 %;
- Credit Card: aproximadamente 28.20 %;
- Installments: aproximadamente 9.19 %.

Sin embargo, los registros afectados contienen un `SK_ID_CURR` válido y no se
observaron inconsistencias entre el cliente asociado al contrato cuando el
`SK_ID_PREV` sí existe.

Por esta razón, estos registros no deben eliminarse.

Para las agregaciones a nivel cliente se utilizará directamente `SK_ID_CURR`
cuando la tabla lo permita.

En `bureau_balance`, los registros sin correspondencia con `bureau` no pueden
asignarse de manera confiable a un cliente.

---

## 14. Prevención de data leakage

La variable `TARGET` se utilizó durante el EDA únicamente para comparar
distribuciones y comportamiento entre clases.

Las features históricas deberán construirse independientemente de `TARGET`.

El pipeline de Feature Engineering deberá:

1. agregar cada fuente histórica por `SK_ID_CURR`;
2. construir las variables sin utilizar `TARGET`;
3. consolidar las fuentes en una tabla por cliente;
4. incorporar `TARGET` únicamente durante la construcción del dataset de
   entrenamiento;
5. realizar las transformaciones aprendidas utilizando únicamente los datos de
   entrenamiento cuando corresponda.

Este principio es consistente con la regla de negocio RN-017 de prevención de
data leakage.

---

## 15. Implicaciones para Feature Engineering

El EDA demuestra que las fuentes históricas contienen señales relevantes y que
deben consolidarse a nivel cliente.

La siguiente fase deberá construir una tabla analítica con granularidad:

`1 fila = 1 SK_ID_CURR`

Las principales familias de features serán:

- características actuales del cliente;
- características financieras;
- antigüedad y estabilidad;
- indicadores de valores faltantes;
- indicadores de valores especiales;
- historial Bureau;
- solicitudes anteriores;
- comportamiento POS;
- utilización de tarjetas;
- comportamiento de cuotas;
- comportamiento histórico de mora;
- indicadores de disponibilidad de cada fuente histórica.

Las tablas históricas deberán agregarse antes de realizar el `join` con la tabla
principal para evitar relaciones uno-a-muchos que multipliquen artificialmente
los clientes.

---

## 16. Conclusión general

El EDA confirma que Home Credit proporciona información suficientemente rica
para construir un modelo de riesgo crediticio utilizando información actual e
histórica del cliente.

Las principales señales observadas se relacionan con:

- comportamiento previo de pagos;
- frecuencia y severidad de mora;
- utilización de crédito;
- aprobación y rechazo de solicitudes anteriores;
- deuda histórica;
- antigüedad del historial;
- disponibilidad o ausencia de información histórica.

También se identificaron desafíos importantes:

- desbalance de clases;
- valores faltantes;
- valores sentinel;
- outliers;
- cobertura desigual entre fuentes;
- brechas de integridad referencial;
- gran volumen de algunas tablas.

Estas condiciones justifican una arquitectura de Feature Engineering basada en
agregaciones por cliente, procesamiento eficiente de grandes volúmenes,
indicadores de cobertura y separación estricta entre construcción de features y
variable objetivo.

Con estas conclusiones se considera finalizada la Fase 7 — Análisis Exploratorio
de Datos y se establecen las bases para iniciar la Fase 8 — Feature Engineering.
