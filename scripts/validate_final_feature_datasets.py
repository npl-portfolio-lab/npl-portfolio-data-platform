from pathlib import Path

from npl_portfolio.core.duckdb_manager import DuckDBManager


ROOT = Path(__file__).resolve().parents[1]

FEATURE_DIR = ROOT / "data" / "processed" / "features"

TRAIN_PATH = FEATURE_DIR / "client_features_train.parquet"
TEST_PATH = FEATURE_DIR / "client_features_test.parquet"

EXPECTED_TRAIN_ROWS = 307_511
EXPECTED_TEST_ROWS = 48_744

HISTORY_FLAGS = [
    "HAS_BUREAU_HISTORY",
    "HAS_BUREAU_BALANCE_HISTORY",
    "HAS_PREVIOUS_APPLICATION_HISTORY",
    "HAS_POS_HISTORY",
    "HAS_CREDIT_CARD_HISTORY",
    "HAS_INSTALLMENTS_HISTORY",
]


def get_columns(connection, path: Path) -> list[str]:
    escaped = str(path.resolve()).replace("'", "''")

    rows = connection.execute(
        f"""
        DESCRIBE
        SELECT *
        FROM read_parquet('{escaped}')
        """
    ).fetchall()

    return [row[0] for row in rows]


def validate_dataset(
    connection,
    path: Path,
    expected_rows: int,
    name: str,
) -> bool:
    escaped = str(path.resolve()).replace("'", "''")

    print()
    print("=" * 80)
    print(f"VALIDANDO {name}")
    print("=" * 80)

    passed = True

    row_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{escaped}')
        """
    ).fetchone()[0]

    if row_count == expected_rows:
        print(f"[PASS] Filas: {row_count:,}")
    else:
        print(
            f"[FAIL] Filas: {row_count:,} "
            f"(esperadas {expected_rows:,})"
        )
        passed = False

    duplicate_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT SK_ID_CURR
            FROM read_parquet('{escaped}')
            GROUP BY SK_ID_CURR
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    if duplicate_count == 0:
        print("[PASS] SK_ID_CURR es único.")
    else:
        print(
            f"[FAIL] Clientes duplicados: "
            f"{duplicate_count:,}"
        )
        passed = False

    null_ids = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{escaped}')
        WHERE SK_ID_CURR IS NULL
        """
    ).fetchone()[0]

    if null_ids == 0:
        print("[PASS] SK_ID_CURR no contiene NULL.")
    else:
        print(
            f"[FAIL] SK_ID_CURR NULL: {null_ids:,}"
        )
        passed = False

    columns = get_columns(
        connection,
        path,
    )

    for flag in HISTORY_FLAGS:
        if flag not in columns:
            print(f"[FAIL] Falta {flag}")
            passed = False
            continue

        invalid_flags = connection.execute(
            f"""
            SELECT COUNT(*)
            FROM read_parquet('{escaped}')
            WHERE {flag} IS NULL
               OR {flag} NOT IN (0, 1)
            """
        ).fetchone()[0]

        if invalid_flags == 0:
            print(f"[PASS] {flag}: valores 0/1 válidos.")
        else:
            print(
                f"[FAIL] {flag}: "
                f"{invalid_flags:,} valores inválidos."
            )
            passed = False

    return passed


def main() -> None:
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(TRAIN_PATH)

    if not TEST_PATH.exists():
        raise FileNotFoundError(TEST_PATH)

    connection = DuckDBManager().connect()

    try:
        train_ok = validate_dataset(
            connection,
            TRAIN_PATH,
            EXPECTED_TRAIN_ROWS,
            "TRAIN",
        )

        test_ok = validate_dataset(
            connection,
            TEST_PATH,
            EXPECTED_TEST_ROWS,
            "TEST",
        )

        train_columns = get_columns(
            connection,
            TRAIN_PATH,
        )

        test_columns = get_columns(
            connection,
            TEST_PATH,
        )

        print()
        print("=" * 80)
        print("VALIDACIONES TRAIN / TEST")
        print("=" * 80)

        if "TARGET" in train_columns:
            print("[PASS] TARGET existe en TRAIN.")
        else:
            print("[FAIL] TARGET no existe en TRAIN.")
            train_ok = False

        if "TARGET" not in test_columns:
            print("[PASS] TARGET no existe en TEST.")
        else:
            print("[FAIL] TARGET aparece en TEST.")
            test_ok = False

        train_without_target = [
            column
            for column in train_columns
            if column != "TARGET"
        ]

        if train_without_target == test_columns:
            print(
                "[PASS] TRAIN y TEST tienen el mismo "
                "schema de features."
            )
        else:
            print(
                "[FAIL] TRAIN y TEST tienen schemas "
                "de features diferentes."
            )

            missing_test = sorted(
                set(train_without_target)
                - set(test_columns)
            )

            extra_test = sorted(
                set(test_columns)
                - set(train_without_target)
            )

            if missing_test:
                print(
                    "       Faltan en TEST:",
                    missing_test,
                )

            if extra_test:
                print(
                    "       Extras en TEST:",
                    extra_test,
                )

            test_ok = False

        target_values = connection.execute(
            f"""
            SELECT DISTINCT TARGET
            FROM read_parquet(
                '{str(TRAIN_PATH.resolve()).replace("'", "''")}'
            )
            ORDER BY TARGET
            """
        ).fetchall()

        target_values = [
            row[0]
            for row in target_values
        ]

        if target_values == [0, 1]:
            print("[PASS] TARGET contiene únicamente 0 y 1.")
        else:
            print(
                f"[FAIL] Valores TARGET: {target_values}"
            )
            train_ok = False

        print()
        print("=" * 80)

        if train_ok and test_ok:
            print("RESULTADO FINAL: PASS")
            print(
                "FASE 8 - FEATURE ENGINEERING "
                "VALIDADA CORRECTAMENTE"
            )
        else:
            print("RESULTADO FINAL: FAIL")
            raise SystemExit(1)

        print("=" * 80)

    finally:
        connection.close()


if __name__ == "__main__":
    main()
