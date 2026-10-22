# Etapa 9 — Persistencia PostgreSQL

## Objetivo

Persistir los datasets consolidados de clientes de Home Credit
en PostgreSQL, manteniendo Parquet como almacenamiento analítico.

## Infraestructura

- PostgreSQL: 17.5
- Base de datos: npl_portfolio_db
- Esquema: npl
- Usuario de aplicación: npl_app
- Conector Python: psycopg 3.3.6

## Tablas

| Tabla | Registros | Columnas | Clave primaria |
|---|---:|---:|---|
| npl.client_features_train | 307511 | 256 | SK_ID_CURR |
| npl.client_features_test | 48744 | 255 | SK_ID_CURR |

TRAIN contiene TARGET; TEST no contiene TARGET.

## Implementación

Script: scripts/persistence/load_client_features.py

- Lectura de Parquet mediante DuckDB.
- Carga a PostgreSQL mediante COPY.
- Procesamiento en lotes de 5000 registros.
- Autenticación mediante contraseña solicitada en ejecución.
- Rechazo de cargas sobre tablas que ya contienen registros.
- Validación de conteos antes de confirmar la transacción.

## Validaciones realizadas

- TRAIN: 307511 registros y 307511 clientes únicos.
- TEST: 48744 registros y 48744 clientes únicos.
- Identificadores nulos: 0 en ambas tablas.
- Clientes compartidos entre TRAIN y TEST en Parquet: 0.
- Clave primaria SK_ID_CURR en ambas tablas.

## Alcance

La persistencia implementada corresponde a los datasets
consolidados de Feature Engineering.

Los datos históricos masivos permanecen en Parquet.

No se modificaron ni reentrenaron los modelos ML V2.

## Pendientes de mejora

- Incorporar pruebas automatizadas de persistencia.
- Validar equivalencia de valores entre Parquet y PostgreSQL.
- Configurar credenciales mediante variables de entorno seguras.
- Verificar la ejecución del script actualizado con to_arrow_reader().
