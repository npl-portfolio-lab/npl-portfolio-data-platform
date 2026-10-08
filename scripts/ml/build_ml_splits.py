from pathlib import Path
from npl_portfolio.core.paths import PROJECT_ROOT

import pandas as pd

from npl_portfolio.core.duckdb_manager import DuckDBManager
from npl_portfolio.ml.dataset_splitter import StratifiedDatasetSplitter


ROOT = PROJECT_ROOT

SOURCE = (
    ROOT
    / "data"
    / "processed"
    / "features"
    / "client_features_train.parquet"
)

OUTPUT = ROOT / "data" / "processed" / "ml"

TRAIN_OUTPUT = OUTPUT / "ml_train.parquet"
VALIDATION_OUTPUT = OUTPUT / "ml_validation.parquet"

VALIDATION_SIZE = 0.20
RANDOM_STATE = 42


def load_split_columns() -> pd.DataFrame:
    """
    Carga únicamente las columnas necesarias para decidir
    la partición de cada cliente.

    Las 256 columnas del dataset NO se materializan en Pandas.
    """
    connection = DuckDBManager().connect()

    try:
        return connection.execute(
            """
            SELECT
                SK_ID_CURR,
                TARGET
            FROM read_parquet(?)
            """,
            [str(SOURCE.resolve())],
        ).fetchdf()

    finally:
        connection.close()


def materialize_partition(
    client_ids: object,
    output_path: Path,
) -> None:
    """
    Materializa una partición usando DuckDB.

    El dataset completo permanece out-of-core.
    """
    ids = pd.DataFrame(
        {
            "SK_ID_CURR": client_ids,
        }
    )

    connection = DuckDBManager().connect()

    try:
        connection.register(
            "selected_clients",
            ids,
        )

        source = str(SOURCE.resolve()).replace("'", "''")
        output = str(output_path.resolve()).replace("'", "''")

        connection.execute(
            f"""
            COPY (
                SELECT source.*
                FROM read_parquet('{source}') AS source

                INNER JOIN selected_clients AS selected
                    ON source.SK_ID_CURR = selected.SK_ID_CURR

                ORDER BY source.SK_ID_CURR
            )
            TO '{output}'
            (
                FORMAT PARQUET,
                COMPRESSION ZSTD
            )
            """
        )

    finally:
        connection.close()


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(
            f"No existe el dataset ML: {SOURCE}"
        )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("FASE 9.2 - SPLIT ESTRATIFICADO")
    print("=" * 80)

    clients = load_split_columns()

    print(
        f"Clientes cargados para split: {len(clients):,}"
    )
    print(
        "Columnas cargadas en Pandas: "
        f"{list(clients.columns)}"
    )

    splitter = StratifiedDatasetSplitter(
        validation_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
    )

    split = splitter.split(clients)

    print(
        f"TRAIN:      {len(split.train_ids):,}"
    )
    print(
        f"VALIDATION: {len(split.validation_ids):,}"
    )

    print()
    print("[OUT-OF-CORE] Materializando TRAIN...")

    materialize_partition(
        split.train_ids,
        TRAIN_OUTPUT,
    )

    print(f"[OK] {TRAIN_OUTPUT}")

    print()
    print("[OUT-OF-CORE] Materializando VALIDATION...")

    materialize_partition(
        split.validation_ids,
        VALIDATION_OUTPUT,
    )

    print(f"[OK] {VALIDATION_OUTPUT}")

    print()
    print("=" * 80)
    print("SPLIT 9.2 FINALIZADO")
    print("=" * 80)


if __name__ == "__main__":
    main()
