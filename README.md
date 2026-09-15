# NPL Portfolio Data Platform

Plataforma de ingeniería de datos, analítica y Machine Learning orientada al
análisis de riesgo de crédito y, posteriormente, a la valoración de portafolios
de cartera.

El proyecto implementa un pipeline completo que abarca adquisición y validación
de datos, análisis exploratorio, construcción de variables, entrenamiento de
modelos, evaluación, valoración financiera, simulación y exposición de
resultados mediante API y dashboard.

---

## 1. Objetivo del proyecto

El objetivo es construir una plataforma modular capaz de transformar grandes
volúmenes de información crediticia en variables analíticas reutilizables para
modelos de riesgo y procesos posteriores de valoración.

La arquitectura busca separar claramente:

- adquisición de datos;
- calidad de datos;
- transformación;
- analítica;
- Feature Engineering;
- Machine Learning;
- evaluación;
- valoración financiera;
- simulación;
- exposición mediante API;
- visualización.

El proyecto se desarrolla utilizando principios de modularidad, separación de
responsabilidades, procesamiento por lotes y trazabilidad.

---

## 2. Dataset actual

La implementación utiliza inicialmente el dataset **Home Credit Default Risk**.

Las principales fuentes procesadas son:

- `application_train`
- `application_test`
- `bureau`
- `bureau_balance`
- `previous_application`
- `POS_CASH_balance`
- `credit_card_balance`
- `installments_payments`

Los archivos CSV originales son transformados a formato Parquet para mejorar
el rendimiento de las consultas analíticas y reducir el costo de lectura de
grandes volúmenes de información.

Algunas tablas superan los 10 millones de registros y `bureau_balance` contiene
más de 27 millones de observaciones mensuales.

---

## 3. Alcance del TARGET

En el dataset Home Credit, `TARGET` representa dificultades de pago asociadas
al riesgo crediticio del solicitante.

Por lo tanto, el modelo construido con este dataset debe interpretarse
principalmente como un modelo de riesgo de incumplimiento o dificultad de pago.

El dataset no contiene directamente información suficiente para entrenar un
modelo real de recuperación de cartera, LGD o flujos de recuperación NPL.

Para las fases posteriores de valoración será necesario separar:

1. estimación del riesgo de crédito;
2. supuestos o modelos de recuperación;
3. flujos de caja esperados;
4. valoración financiera del portafolio.

Esta separación evita atribuir a `TARGET` un significado que el dataset
original no proporciona.

---

## 4. Arquitectura general

```text
                  FUENTES DE DATOS
                        |
                        v
                CSV / datos externos
                        |
                        v
                 INGESTION LAYER
                        |
                        v
               DATA QUALITY LAYER
                        |
                        v
               TRANSFORMATION LAYER
                        |
                        v
                     PARQUET
                        |
                        v
                 ANALYTICS LAYER
                        |
                        v
               FEATURE ENGINEERING
                        |
                        v
                 MACHINE LEARNING
                        |
                        v
                    EVALUATION
                        |
                        v
             FINANCIAL VALUATION
                        |
                        v
                MONTE CARLO
                        |
                        v
                    FASTAPI
                        |
                        v
               REACT DASHBOARD


La lógica de negocio y analítica se mantiene fuera de notebooks y scripts
siempre que sea posible para permitir su reutilización desde pipelines,
servicios, APIs y futuras interfaces.

5. Estructura del proyecto
npl-portfolio-data-platform/
|
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── external/
|
├── docs/
|
├── notebooks/
|
├── scripts/
|
├── src/
│   └── npl_portfolio/
│       ├── core/
│       ├── domain/
│       ├── ingestion/
│       ├── profiling/
│       ├── validation/
│       ├── data_quality/
│       ├── transformation/
│       ├── persistence/
│       ├── pipelines/
│       ├── analytics/
│       ├── features/
│       ├── ml/
│       ├── evaluation/
│       ├── valuation/
│       └── simulation/
|
├── tests/
├── requirements.txt
└── README.md
6. Flujo de datos implementado

Actualmente el pipeline principal sigue este flujo:

Home Credit CSV
       |
       v
BatchCSVReader
       |
       +--------------------+
       |                    |
       v                    v
   Profiling           Data Quality
       |                    |
       +---------+----------+
                 |
                 v
          Transformation
                 |
                 v
              Parquet
                 |
                 v
               EDA
                 |
                 v
      Historical Analytics
                 |
                 v
      Client Aggregations

Para archivos CSV grandes se utiliza lectura por lotes para evitar cargar
innecesariamente datasets completos en memoria.

Para las tablas históricas de mayor tamaño se utiliza DuckDB sobre archivos
Parquet.

7. Calidad de datos

La plataforma incluye controles de calidad estructurales, semánticos y
referenciales.

Entre los controles implementados se encuentran:

columnas obligatorias;
valores nulos;
duplicados;
unicidad de claves;
valores permitidos;
valores positivos y no negativos;
integridad referencial entre tablas.

Los controles estructurales y semánticos principales fueron superados por los
datasets procesados.

También se identificaron diferencias de cobertura entre algunas tablas
históricas.

Por ejemplo, existen registros históricos cuyo SK_ID_PREV no está presente
en previous_application.

Estos registros no se eliminan automáticamente cuando todavía pueden asociarse
directamente con un cliente mediante SK_ID_CURR.

En bureau_balance existe además una parte del historial cuyo
SK_ID_BUREAU no está disponible en bureau. Estos registros se conservan en
la fuente procesada, pero no pueden utilizarse para agregaciones por cliente
cuando no existe una relación hacia SK_ID_CURR.

8. Transformación a Parquet

Las ocho tablas principales fueron convertidas desde CSV a Parquet:

Dataset	Registros
application_train	307,511
application_test	48,744
bureau	1,716,428
bureau_balance	27,299,925
previous_application	1,670,214
POS_CASH_balance	10,001,358
credit_card_balance	3,840,312
installments_payments	13,605,401

La transformación fue validada comparando:

cantidad de registros;
cantidad de columnas;
nombres y orden de columnas.

Los archivos resultantes se almacenan en:

data/processed/home_credit/
9. Análisis exploratorio

El EDA se divide en dos componentes principales.

Application data

Se analizó:

distribución de TARGET;
valores faltantes;
variables numéricas;
valores especiales;
variables categóricas;
relaciones descriptivas entre variables y TARGET.

TARGET presenta un desbalance importante:

TARGET = 0    91.93%
TARGET = 1     8.07%

Este desbalance deberá considerarse durante entrenamiento y evaluación.

Historical data

Se analizaron las tablas:

bureau;
previous_application;
POS_CASH_balance;
credit_card_balance;
installments_payments;
bureau_balance.

Las tablas históricas se agregan conceptualmente por SK_ID_CURR antes de
compararlas con TARGET.

Esto permite analizar el comportamiento histórico del cliente sin incorporar
la variable objetivo dentro de la construcción de las variables.

10. Prevención de Data Leakage

Una regla central del proyecto es evitar que TARGET participe en la
construcción de variables predictoras.

El patrón utilizado es:

HISTORICAL DATA
      |
      v
aggregate by SK_ID_CURR
      |
      v
CLIENT FEATURES
      |
      v
join TARGET
      |
      v
EDA / TRAINING / EVALUATION

y no:

TARGET
   |
   v
feature construction

Las agregaciones históricas deben poder generarse independientemente de la
existencia de TARGET.

11. Estado de implementación
Fase	Descripción	Estado
1	Requerimientos	Completada
2	Reglas de negocio, casos de uso y trazabilidad	Completada
3	Arquitectura	Completada
4	Modelo lógico de datos	Completada
5	Selección y adquisición del dataset	Completada
6	Ingeniería de datos y EDA	Completada
7	Agregaciones por cliente	En desarrollo
8	Feature Engineering	Pendiente
9	Machine Learning	Pendiente
10	Evaluación y backtesting	Pendiente
11	Valoración financiera	Pendiente
12	Simulación Monte Carlo	Pendiente
13	FastAPI	Pendiente
14	Docker, pruebas y observabilidad	Pendiente
15	Documentación final y publicación	Pendiente
12. Etapa actual
Fase 7 — Agregaciones por cliente

El siguiente objetivo es transformar las diferentes tablas históricas en una
representación consolidada con una única fila por SK_ID_CURR.

Conceptualmente:

application
    |
    +--- bureau
    |
    +--- previous_application
    |
    +--- POS_CASH_balance
    |
    +--- credit_card_balance
    |
    +--- installments_payments
    |
    +--- bureau_balance
    |
    v
CLIENT HISTORICAL FEATURES

La salida prevista será un dataset Parquet reutilizable por las siguientes
capas:

data/processed/features/
    client_historical_features.parquet

Esta tabla no utilizará TARGET para construir sus variables.

13. Tecnologías

El proyecto utiliza principalmente:

Python 3.11
Pandas
PyArrow
DuckDB
Parquet
Pytest
Ruff
Git
GitHub

En fases posteriores se incorporarán tecnologías para Machine Learning,
FastAPI, contenedores, observabilidad y frontend.

14. Principios de diseño

El desarrollo sigue los siguientes principios:

separación de responsabilidades;
componentes reutilizables;
procesamiento eficiente de grandes datasets;
conservación de datos originales;
trazabilidad de transformaciones;
controles explícitos de calidad;
prevención de data leakage;
agregaciones independientes de TARGET;
arquitectura preparada para API y frontend;
reproducibilidad de los pipelines.
15. Roadmap
Requirements
     ✓
Business Rules
     ✓
Architecture
     ✓
Logical Data Model
     ✓
Dataset Acquisition
     ✓
Data Engineering
     ✓
EDA
     ✓
Client Aggregations
     ◄ CURRENT
Feature Engineering
     ↓
Machine Learning
     ↓
Evaluation / Backtesting
     ↓
Financial Valuation
     ↓
Monte Carlo
     ↓
FastAPI
     ↓
React Dashboard
     ↓
Docker / Observability
16. Reproducibilidad

Los datasets originales y procesados no se almacenan directamente en Git
debido a su tamaño.

La estructura del repositorio conserva los directorios necesarios mediante
archivos .gitkeep, mientras que los datos son excluidos mediante
.gitignore.

Los scripts y servicios permiten reproducir progresivamente las diferentes
etapas del pipeline a partir de las fuentes originales.

17. Estado del proyecto

En desarrollo activo.

Actualmente se encuentra finalizada la etapa de análisis exploratorio y se está
implementando la consolidación de variables históricas por cliente.

El siguiente entregable técnico es:

client_historical_features.parquet

que servirá como entrada para la etapa de Feature Engineering.


### Una corrección importante respecto a nuestro roadmap

Aquí deliberadamente agrupé **Ingeniería de Datos + EDA como Fase 6** porque en nuestra implementación ya hicimos ambas antes de las agregaciones. Pero en la planificación original teníamos:

```text
Fase 5  Dataset
Fase 6  Ingeniería de datos / ETL
Fase 7  EDA
Fase 8  Feature Engineering
...

Y acabamos llamando coloquialmente a lo siguiente “Etapa 7 — agregaciones”.

Para que la documentación profesional no tenga dos numeraciones distintas, te recomiendo que no cambiemos el roadmap original. Entonces antes de hacer commit yo corregiría esa parte del README a:

Fase 5  Selección/adquisición dataset       COMPLETADA
Fase 6  Ingeniería de datos / ETL           COMPLETADA
Fase 7  EDA                                 COMPLETADA
Fase 8  Feature Engineering                 SIGUIENTE
Fase 9  Machine Learning
Fase 10 Evaluación/backtesting
Fase 11 Valoración financiera
Fase 12 Monte Carlo
Fase 13 FastAPI
Fase 14 Docker/pruebas/observabilidad
Fase 15 Documentación/GitHub