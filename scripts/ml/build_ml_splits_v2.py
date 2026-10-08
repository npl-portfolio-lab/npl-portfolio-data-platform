"""Fase 10.6: particiones estratificadas V2, sin sobrescrituras."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from npl_portfolio.core.paths import PROJECT_ROOT, ML_DATA_DIR
from npl_portfolio.core.duckdb_manager import DuckDBManager
from npl_portfolio.ml.three_way_splitter import StratifiedThreeWaySplitter


SOURCE = (
    PROJECT_ROOT
    / "data/processed/features/client_features_train.parquet"
)

OUTPUT = ML_DATA_DIR / "v2"
SPLITS = ("train", "validation", "test")
RANDOM_STATE = 42


def load_clients():
    connection = DuckDBManager().connect()
    try:
        return connection.execute(
            """
            SELECT SK_ID_CURR, TARGET
            FROM read_parquet(?)
            """,
            [str(SOURCE.resolve())],
        ).fetchdf()
    finally:
        connection.close()


def validate_split(clients, split):
    all_ids = set(clients["SK_ID_CURR"].tolist())

    partitions = {
        "train": set(split.train_ids.tolist()),
        "validation": set(split.validation_ids.tolist()),
        "test": set(split.test_ids.tolist()),
    }

    for name, ids in partitions.items():
        expected = len(getattr(split, f"{name}_ids"))
        if len(ids) != expected:
            raise ValueError(f"Identificadores duplicados en {name}")

    if not partitions["train"].isdisjoint(partitions["validation"]):
        raise ValueError("Solapamiento TRAIN/VALIDATION")

    if not partitions["train"].isdisjoint(partitions["test"]):
        raise ValueError("Solapamiento TRAIN/TEST")

    if not partitions["validation"].isdisjoint(partitions["test"]):
        raise ValueError("Solapamiento VALIDATION/TEST")

    if set.union(*partitions.values()) != all_ids:
        raise ValueError("Las particiones no cubren todos los clientes")


def materialize(client_ids, destination):
    ids = pd.DataFrame({"SK_ID_CURR": client_ids})

    connection = DuckDBManager().connect()

    try:
        connection.register("selected_clients", ids)

        source = str(SOURCE.resolve()).replace("'", "''")
        output = str(destination.resolve()).replace("'", "''")

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
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
    finally:
        connection.close()


def main():
    print("=" * 70)
    print("FASE 10.6 - MATERIALIZACION DE PARTICIONES V2")
    print("=" * 70)

    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)

    destinations = {
        name: OUTPUT / f"ml_{name}.parquet"
        for name in SPLITS
    }
    manifest_path = OUTPUT / "split_manifest.json"

    # Política de no sobrescritura: comprobar antes de generar datos.
    existing = [
        str(path)
        for path in (*destinations.values(), manifest_path)
        if path.exists()
    ]

    if existing:
        raise FileExistsError(
            "La salida V2 ya existe. No se sobrescribirá:\n"
            + "\n".join(existing)
        )

    clients = load_clients()

    splitter = StratifiedThreeWaySplitter(
        train_size=0.64,
        validation_size=0.16,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    split = splitter.split(clients)
    validate_split(clients, split)

    OUTPUT.mkdir(parents=True, exist_ok=True)

    manifest = {
        "phase": "10.6",
        "version": "v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(SOURCE.relative_to(PROJECT_ROOT)),
        "random_state": RANDOM_STATE,
        "proportions": {
            "train": 0.64,
            "validation": 0.16,
            "test": 0.20,
        },
        "partitions": {},
    }

    for name in SPLITS:
        ids = getattr(split, f"{name}_ids")
        destination = destinations[name]

        print(f"\n[BUILD] {name.upper()}: {len(ids):,} clientes")

        materialize(ids, destination)

        selected = clients[
            clients["SK_ID_CURR"].isin(ids)
        ]

        manifest["partitions"][name] = {
            "rows": int(len(ids)),
            "positives": int((selected["TARGET"] == 1).sum()),
            "negatives": int((selected["TARGET"] == 0).sum()),
            "prevalence": float(selected["TARGET"].mean()),
            "path": str(destination.relative_to(PROJECT_ROOT)),
        }

        print(f"[OK] {destination}")

    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)

    print(f"\n[OK] Manifiesto: {manifest_path}")
    print("[OK] PARTICIONES V2 GENERADAS")
    print("[INFO] TEST permanece reservado para evaluación final.")


if __name__ == "__main__":
    main()
