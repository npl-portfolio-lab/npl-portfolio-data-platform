"""Fase 10.6: validación de integridad de particiones V2."""

import json

from npl_portfolio.core.paths import ML_DATA_DIR
from npl_portfolio.core.duckdb_manager import DuckDBManager


ROOT = ML_DATA_DIR / "v2"
MANIFEST = ROOT / "split_manifest.json"
SPLITS = ("train", "validation", "test")


def main():
    print("=" * 70)
    print("FASE 10.6 - VALIDACION DE PARTICIONES V2")
    print("=" * 70)

    if not MANIFEST.is_file():
        raise FileNotFoundError(MANIFEST)

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    if manifest.get("phase") != "10.6":
        raise ValueError("Manifiesto incompatible")

    connection = DuckDBManager().connect()

    try:
        results = {}
        schemas = {}

        for name in SPLITS:
            path = ROOT / f"ml_{name}.parquet"

            if not path.is_file():
                raise FileNotFoundError(path)

            connection.execute(
                f"CREATE TEMP VIEW {name} AS "
                "SELECT * FROM read_parquet("
                f"'{str(path.resolve()).replace(chr(39), chr(39) * 2)}'"
                ")"
            )

            columns = connection.execute(
                f"DESCRIBE SELECT * FROM {name}"
            ).fetchall()

            schemas[name] = [
                (row[0], row[1]) for row in columns
            ]

            row = connection.execute(
                f"""
                SELECT
                    COUNT(*),
                    COUNT(DISTINCT SK_ID_CURR),
                    COUNT(*) FILTER (WHERE TARGET = 1),
                    COUNT(*) FILTER (WHERE TARGET = 0),
                    COUNT(*) FILTER (
                        WHERE SK_ID_CURR IS NULL OR TARGET IS NULL
                    ),
                    COUNT(*) FILTER (
                        WHERE TARGET NOT IN (0, 1)
                    )
                FROM {name}
                """
            ).fetchone()

            total, unique, positives, negatives, nulls, invalid = row

            expected = manifest["partitions"][name]

            assert total == expected["rows"], name
            assert positives == expected["positives"], name
            assert negatives == expected["negatives"], name
            assert total == unique, f"Duplicados en {name}"
            assert nulls == 0, f"Nulos en {name}"
            assert invalid == 0, f"TARGET inválido en {name}"
            assert positives + negatives == total, name

            results[name] = total

            print(
                f"[PASS] {name.upper():10} "
                f"filas={total:,} "
                f"positivos={positives:,} "
                f"negativos={negatives:,}"
            )

        assert schemas["train"] == schemas["validation"]
        assert schemas["train"] == schemas["test"]

        print("[PASS] Esquemas compatibles")

        for first, second in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        ):
            overlap = connection.execute(
                f"""
                SELECT COUNT(*)
                FROM {first} a
                INNER JOIN {second} b
                    ON a.SK_ID_CURR = b.SK_ID_CURR
                """
            ).fetchone()[0]

            assert overlap == 0, (
                f"Clientes compartidos entre {first} y {second}"
            )

            print(
                f"[PASS] {first.upper()} / {second.upper()}: "
                "0 clientes compartidos"
            )

        total = sum(results.values())

        assert total == 307_511, (
            f"Total inesperado: {total}"
        )

        print(f"\n[PASS] Total registros: {total:,}")
        print("[OK] PARTICIONES V2 VALIDADAS")

    finally:
        connection.close()


if __name__ == "__main__":
    main()
