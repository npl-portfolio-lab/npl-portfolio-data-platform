"""Genera documentación reproducible de la Fase 10.6."""

import json
from datetime import datetime, timezone

from npl_portfolio.core.paths import (
    PROJECT_ROOT,
    ML_DATA_DIR,
    ML_ARTIFACTS_DIR,
)


DATA_DIR = ML_DATA_DIR / "v2"
SPLIT_MANIFEST = DATA_DIR / "split_manifest.json"
PREPROCESSING_MANIFEST = DATA_DIR / "preprocessing_manifest.json"
CONTRACT_PATH = (
    ML_ARTIFACTS_DIR / "v2" / "preprocessing_contract.joblib"
)

OUTPUT = (
    PROJECT_ROOT
    / "docs/ml/fase_10_6_protocolo_experimental.md"
)

SPLITS = ("train", "validation", "test")


def load_json(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    return json.loads(path.read_text(encoding="utf-8"))


def format_number(value):
    return f"{value:,}".replace(",", ".")


def main():
    split_data = load_json(SPLIT_MANIFEST)
    preprocessing_data = load_json(PREPROCESSING_MANIFEST)

    if split_data.get("version") != "v2":
        raise ValueError("Versión de particiones incorrecta")

    if preprocessing_data.get("version") != "v2":
        raise ValueError("Versión de preprocesamiento incorrecta")

    if not CONTRACT_PATH.is_file():
        raise FileNotFoundError(CONTRACT_PATH)

    rows = []
    total_rows = 0
    total_positive = 0
    total_negative = 0
    total_batches = 0

    for name in SPLITS:
        partition = split_data["partitions"][name]
        processed = preprocessing_data["partitions"][name]

        count = partition["rows"]
        positive = partition["positives"]
        negative = partition["negatives"]
        batches = processed["batches"]

        if count != processed["rows"]:
            raise ValueError(
                f"Conteos inconsistentes en {name}"
            )

        if processed["features"] != preprocessing_data["features"]:
            raise ValueError(
                f"Variables inconsistentes en {name}"
            )

        if positive + negative != count:
            raise ValueError(
                f"Etiquetas inconsistentes en {name}"
            )

        total_rows += count
        total_positive += positive
        total_negative += negative
        total_batches += batches

        rows.append(
            f"| {name.upper()} "
            f"| {format_number(count)} "
            f"| {format_number(positive)} "
            f"| {format_number(negative)} "
            f"| {batches} |"
        )

    proportions = split_data["proportions"]

    generated_at = datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%d %H:%M UTC")

    content = f"""# Fase 10.6 — Protocolo experimental V2

## 1. Objetivo

Construir un protocolo experimental reproducible con particiones
TRAIN, VALIDATION y TEST, manteniendo separados los artefactos
de experimentos anteriores.

## 2. Origen de datos

- Fuente: `{split_data["source"]}`
- Total de registros: {format_number(total_rows)}
- Identificador: `SK_ID_CURR`
- Variable objetivo: `TARGET`
- Semilla aleatoria: {split_data["random_state"]}

En Home Credit, `TARGET=1` representa dificultad de pago;
no equivale directamente a recuperación de cartera castigada.

## 3. Distribución

| Partición | Registros | TARGET=1 | TARGET=0 | Lotes |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}
| **TOTAL** | **{format_number(total_rows)}** | **{format_number(total_positive)}** | **{format_number(total_negative)}** | **{total_batches}** |

Proporciones configuradas:

- TRAIN: {proportions["train"]:.0%}
- VALIDATION: {proportions["validation"]:.0%}
- TEST: {proportions["test"]:.0%}

## 4. Preprocesamiento

Contrato construido exclusivamente desde TRAIN V2.

- Columnas numéricas: {preprocessing_data["numeric_columns"]}
- Columnas categóricas: {preprocessing_data["categorical_columns"]}
- Variables resultantes: {preprocessing_data["features"]}
- Tamaño máximo de lote: {preprocessing_data["batch_size"]}
- Contrato: `data/artifacts/ml/v2/preprocessing_contract.joblib`

Se utiliza imputación mediante estadísticas de TRAIN y
codificación OneHotEncoder con categorías aprendidas de TRAIN.

VALIDATION y TEST se transforman utilizando el mismo contrato.

## 5. Validaciones

Los scripts de auditoría comprueban:

- Integridad de identificadores y etiquetas.
- Ausencia de clientes compartidos entre particiones.
- Consistencia de esquemas Parquet.
- Coincidencia de conteos con los manifiestos.
- Dimensiones de matrices dispersas CSR.
- Valores finitos almacenados en las matrices.
- Consistencia de metadatos y conteos positivos.

La ejecución satisfactoria de las auditorías debe verificarse
por separado. Este documento se genera desde los manifiestos,
no desde un registro persistente de resultados de auditoría.

## 6. Independencia experimental

TRAIN se utiliza para aprender estadísticas y entrenar modelos.

VALIDATION se reserva para comparación y selección.

TEST se reserva para evaluación predictiva final y no debe
utilizarse para seleccionar modelos ni umbrales.

**Limitación:** TEST V2 contiene clientes procedentes del
conjunto etiquetado utilizado en experimentos anteriores.
Por ello no constituye una evaluación externa completamente
inédita respecto a todo el historial del proyecto.

## 7. Limitaciones pendientes

- Auditar disponibilidad temporal de las variables originales.
- Verificar posibles fugas de información en feature engineering.
- Incorporar una política para columnas completamente nulas.
- No se ha realizado comparación fila por fila de valores
  transformados contra el origen.
- No se han calculado métricas predictivas sobre TEST V2.

## 8. Artefactos

- `data/processed/ml/v2/`
- `data/artifacts/ml/v2/`
- `scripts/ml/build_ml_splits_v2.py`
- `scripts/ml/validate_ml_splits_v2.py`
- `scripts/ml/build_ml_preprocessed_v2.py`
- `scripts/ml/audit_preprocessed_v2.py`
- `src/npl_portfolio/ml/three_way_splitter.py`

## 9. Siguiente fase

Fase 10.7: entrenamiento y comparación de modelos V2,
utilizando TRAIN y VALIDATION.

---

Documento generado automáticamente: {generated_at}.
"""

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(content, encoding="utf-8")

    print("=" * 70)
    print("FASE 10.6 - DOCUMENTACION AUTOMATICA")
    print("=" * 70)
    print(f"[OK] Registros documentados: {format_number(total_rows)}")
    print(f"[OK] Lotes documentados: {total_batches}")
    print(f"[OK] Variables documentadas: {preprocessing_data['features']}")
    print(f"[OK] Documento generado: {OUTPUT}")


if __name__ == "__main__":
    main()
