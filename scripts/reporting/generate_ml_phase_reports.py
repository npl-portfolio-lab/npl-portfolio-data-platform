from datetime import datetime, timezone
from pathlib import Path
from npl_portfolio.core.paths import PROJECT_ROOT, ML_DATA_DIR, ML_ARTIFACTS_DIR, DOCS_DIR as DOCS_ROOT

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from npl_portfolio.core.duckdb_manager import DuckDBManager

ROOT = PROJECT_ROOT

ML_DIR = ML_DATA_DIR
ARTIFACTS_DIR = ML_ARTIFACTS_DIR
DOCS_DIR = DOCS_ROOT / "machine_learning"

SOURCE = ROOT / "data" / "processed" / "features" / "client_features_train.parquet"
TRAIN = ML_DIR / "ml_train.parquet"
VALIDATION = ML_DIR / "ml_validation.parquet"
CONTRACT = ARTIFACTS_DIR / "preprocessing_contract.joblib"


def write_report(filename: str, content: str) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    output = DOCS_DIR / filename
    temporary = output.with_suffix(".md.tmp")

    temporary.write_text(content.strip() + "\n", encoding="utf-8")
    temporary.replace(output)

    print(f"[OK] Informe generado: {output}")


def header(phase: str, title: str) -> str:
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return f"# Fase {phase} — {title}\n\n" f"**Informe generado:** {date}\n\n"


def query_stats(connection, path: Path) -> dict:
    result = connection.execute(
        """
        SELECT
            COUNT(*) AS total,
            COUNT(DISTINCT SK_ID_CURR) AS unique_ids,
            SUM(CASE WHEN TARGET = 1 THEN 1 ELSE 0 END) AS positives,
            AVG(TARGET) AS target_rate
        FROM read_parquet(?)
        """,
        [str(path.resolve())],
    ).fetchone()

    return {
        "rows": int(result[0]),
        "unique_ids": int(result[1]),
        "positives": int(result[2]),
        "target_rate": float(result[3]),
    }


def generate_phase_9_1() -> None:
    print("\n[9.1] Contrato de preprocesamiento")

    if not CONTRACT.exists():
        raise FileNotFoundError(CONTRACT)

    contract = joblib.load(CONTRACT)

    numeric = len(contract.numeric_medians)
    categorical = len(contract.categorical_modes)
    vocabularies = len(contract.categorical_values)

    if numeric == 0 and categorical == 0:
        raise ValueError("El contrato no contiene estadísticas.")

    content = header("9.1", "Contrato de preprocesamiento")

    content += f"""## 1. Objetivo

Definir y conservar las estadísticas necesarias para transformar
los datos de entrenamiento y validación de manera consistente,
evitando aprender información del conjunto VALIDATION.

## 2. Metodología

Se utiliza `PreprocessingContractBuilder` y DuckDB para calcular
las estadísticas a partir del archivo TRAIN.

- Variables numéricas: mediana para imputación.
- Variables categóricas: moda para imputación.
- Categorías: valores distintos registrados durante el ajuste.

El contrato se almacena con Joblib para reutilizarlo.

## 3. Resultados registrados

| Indicador | Resultado |
|---|---:|
| Variables con mediana | {numeric:,} |
| Variables con moda | {categorical:,} |
| Variables con vocabulario | {vocabularies:,} |
| Archivo del contrato | `preprocessing_contract.joblib` |

## 4. Validaciones

- El archivo del contrato existe y se puede cargar.
- Contiene estadísticas de preprocesamiento.
- Se verificaron las cantidades registradas en el contrato.

## 5. Conclusión

El contrato permite aplicar estadísticas consistentes a los
conjuntos de datos. Su implementación está diseñada para aprender
exclusivamente de TRAIN.

Esta revisión confirma la existencia y estructura del artefacto;
no demuestra por sí sola el origen histórico de cada estadística.
"""

    write_report("phase_9_1_ml_contract.md", content)


def generate_phase_9_2() -> None:
    print("\n[9.2] División estratificada")

    for path in (SOURCE, TRAIN, VALIDATION):
        if not path.exists():
            raise FileNotFoundError(path)

    connection = DuckDBManager().connect()

    try:
        source = query_stats(connection, SOURCE)
        train = query_stats(connection, TRAIN)
        validation = query_stats(connection, VALIDATION)

        overlap = connection.execute(
            """
            SELECT COUNT(*)
            FROM read_parquet(?) AS t
            INNER JOIN read_parquet(?) AS v
                USING (SK_ID_CURR)
            """,
            [str(TRAIN.resolve()), str(VALIDATION.resolve())],
        ).fetchone()[0]

        source_schema = connection.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)",
            [str(SOURCE.resolve())],
        ).fetchall()

        train_schema = connection.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)",
            [str(TRAIN.resolve())],
        ).fetchall()

        validation_schema = connection.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)",
            [str(VALIDATION.resolve())],
        ).fetchall()

    finally:
        connection.close()

    checks = {
        "Cobertura completa": train["rows"] + validation["rows"] == source["rows"],
        "IDs únicos en TRAIN": train["rows"] == train["unique_ids"],
        "IDs únicos en VALIDATION": validation["rows"] == validation["unique_ids"],
        "Sin clientes compartidos": overlap == 0,
        "Schemas idénticos": source_schema == train_schema == validation_schema,
        "Estratificación conservada": abs(
            train["target_rate"] - validation["target_rate"]
        )
        < 0.001,
    }

    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"Validaciones 9.2 fallidas: {failed}")

    validation_pct = validation["rows"] / source["rows"] * 100

    content = header("9.2", "División estratificada TRAIN / VALIDATION")

    content += f"""## 1. Objetivo

Separar los clientes en conjuntos independientes para entrenar
y evaluar los modelos, manteniendo una distribución comparable
de la variable objetivo `TARGET`.

## 2. Metodología

Se utiliza `StratifiedDatasetSplitter` con:

- TRAIN: 80 %.
- VALIDATION: 20 %.
- Semilla aleatoria: `42`.
- Identificador de cliente: `SK_ID_CURR`.

DuckDB permite materializar las particiones en Parquet sin
cargar todas las variables en memoria.

## 3. Resultados

| Indicador | SOURCE | TRAIN | VALIDATION |
|---|---:|---:|---:|
| Registros | {source['rows']:,} | {train['rows']:,} | {validation['rows']:,} |
| Clientes únicos | {source['unique_ids']:,} | {train['unique_ids']:,} | {validation['unique_ids']:,} |
| Positivos TARGET=1 | {source['positives']:,} | {train['positives']:,} | {validation['positives']:,} |
| Tasa positiva | {source['target_rate']:.4%} | {train['target_rate']:.4%} | {validation['target_rate']:.4%} |

Porcentaje real de VALIDATION: **{validation_pct:.2f}%**.

## 4. Validaciones

| Validación | Resultado |
|---|---|
""" + "\n".join(
        f"| {name} | {'PASS' if passed else 'FAIL'} |"
        for name, passed in checks.items()
    )

    content += f"""

Clientes compartidos entre TRAIN y VALIDATION: **{overlap:,}**.

## 5. Conclusión

Las particiones cumplen las verificaciones de cobertura,
unicidad, separación de clientes, consistencia de esquema
y conservación de la distribución de TARGET.

Esta validación no descarta por sí sola otras formas
de fuga de información en las variables predictoras.
"""

    write_report("phase_9_2_data_split.md", content)


def inspect_batches(prefix: str) -> dict:
    matrices = sorted(ML_DIR.glob(f"{prefix}_X_*.npz"))
    metadata = sorted(ML_DIR.glob(f"{prefix}_meta_*.parquet"))

    if not matrices or len(matrices) != len(metadata):
        raise AssertionError(
            f"{prefix}: matrices y metadatos faltantes o inconsistentes."
        )

    expected_names = [f"{prefix}_X_{i:03d}.npz" for i in range(1, len(matrices) + 1)]

    if [path.name for path in matrices] != expected_names:
        raise AssertionError(f"{prefix}: numeración de lotes inválida.")

    total_rows = 0
    n_features = None
    nonzeros = 0
    all_ids = set()

    for matrix_path, metadata_path in zip(matrices, metadata, strict=True):
        matrix = sparse.load_npz(matrix_path)
        meta = pd.read_parquet(metadata_path)

        if not sparse.isspmatrix_csr(matrix):
            raise AssertionError(f"{matrix_path.name}: no es CSR.")

        if matrix.shape[0] != len(meta):
            raise AssertionError(f"{matrix_path.name}: filas inconsistentes.")

        if n_features is None:
            n_features = matrix.shape[1]
        elif matrix.shape[1] != n_features:
            raise AssertionError(f"{prefix}: features inconsistentes.")

        if not np.isfinite(matrix.data).all():
            raise AssertionError(f"{matrix_path.name}: valores no finitos.")

        if set(meta.columns) != {"SK_ID_CURR", "TARGET"}:
            raise AssertionError(f"{metadata_path.name}: metadata inválida.")

        if meta.isna().any().any():
            raise AssertionError(f"{metadata_path.name}: valores nulos.")

        if not meta["TARGET"].isin([0, 1]).all():
            raise AssertionError(f"{metadata_path.name}: TARGET inválido.")

        ids = set(meta["SK_ID_CURR"].astype(int))

        if len(ids) != len(meta) or not all_ids.isdisjoint(ids):
            raise AssertionError(f"{prefix}: IDs duplicados.")

        all_ids.update(ids)
        total_rows += matrix.shape[0]
        nonzeros += matrix.nnz

        print(f"[PASS] {matrix_path.name}")

    return {
        "rows": total_rows,
        "features": n_features,
        "batches": len(matrices),
        "nonzeros": nonzeros,
        "ids": all_ids,
    }


def generate_phase_9_3() -> None:
    print("\n[9.3] Preprocesamiento sparse")

    if not CONTRACT.exists():
        raise FileNotFoundError(CONTRACT)

    contract = joblib.load(CONTRACT)

    train = inspect_batches("train")
    validation = inspect_batches("validation")

    if train["rows"] != 246_008:
        raise AssertionError("Cantidad TRAIN inesperada.")

    if validation["rows"] != 61_503:
        raise AssertionError("Cantidad VALIDATION inesperada.")

    if train["features"] != validation["features"]:
        raise AssertionError("Diferente número de features.")

    if not train["ids"].isdisjoint(validation["ids"]):
        raise AssertionError("TRAIN y VALIDATION comparten clientes.")

    content = header("9.3", "Preprocesamiento y matrices sparse")

    content += f"""## 1. Objetivo

Transformar las variables originales en matrices numéricas
adecuadas para entrenar modelos de Machine Learning,
controlando el consumo de memoria.

## 2. Metodología

El preprocesamiento utiliza:

- DuckDB para lectura de archivos Parquet.
- Procesamiento por lotes de 20.000 registros.
- Estadísticas de imputación aprendidas desde TRAIN.
- Codificación de variables categóricas.
- Matrices dispersas CSR almacenadas como archivos NPZ.
- Metadatos de cliente y TARGET almacenados en Parquet.

## 3. Resultados

| Indicador | TRAIN | VALIDATION |
|---|---:|---:|
| Registros | {train['rows']:,} | {validation['rows']:,} |
| Features transformadas | {train['features']:,} | {validation['features']:,} |
| Lotes | {train['batches']} | {validation['batches']} |
| Elementos almacenados (nnz) | {train['nonzeros']:,} | {validation['nonzeros']:,} |
| Formato | CSR | CSR |

El contrato registra:

- **{len(contract.numeric_medians)}** medianas numéricas.
- **{len(contract.categorical_modes)}** modas categóricas.
- **{len(contract.categorical_values)}** vocabularios categóricos.

## 4. Validaciones

| Comprobación | Resultado |
|---|---|
| Matrices y metadatos correspondientes | PASS |
| Formato CSR | PASS |
| Número consistente de features | PASS |
| Ausencia de NaN e infinitos | PASS |
| TARGET binario y sin nulos | PASS |
| IDs únicos por conjunto | PASS |
| Sin clientes compartidos | PASS |
| Cantidades esperadas de registros | PASS |

## 5. Conclusión

Los conjuntos transformados cumplen las comprobaciones
estructurales y de integridad realizadas.

El procesamiento por lotes y el formato sparse permiten
evitar materializar toda la matriz como un arreglo denso.

La evaluación de la calidad predictiva corresponde
a las fases posteriores de Machine Learning.
"""

    write_report("phase_9_3_preprocessing.md", content)


def main() -> None:
    print("=" * 70)
    print("GENERACIÓN DE DOCUMENTACIÓN ML — FASES 9.1, 9.2 Y 9.3")
    print("=" * 70)

    generate_phase_9_1()
    generate_phase_9_2()
    generate_phase_9_3()

    print("\n[OK] Documentación generada correctamente.")


if __name__ == "__main__":
    main()
