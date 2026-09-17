from pathlib import Path

from npl_portfolio.core.duckdb_manager import DuckDBManager


ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "data" / "processed" / "home_credit"
FEATURES = ROOT / "data" / "interim" / "features"
OUTPUT = ROOT / "data" / "processed" / "features"

OUTPUT.mkdir(parents=True, exist_ok=True)


HISTORICAL_FEATURES = [
    ("bureau", FEATURES / "bureau_features.parquet"),
    (
        "bureau_balance",
        FEATURES / "bureau_balance_features.parquet",
    ),
    (
        "previous_application",
        FEATURES / "previous_application_features.parquet",
    ),
    (
        "pos_cash",
        FEATURES / "pos_cash_features.parquet",
    ),
    (
        "credit_card",
        FEATURES / "credit_card_features.parquet",
    ),
    (
        "installments",
        FEATURES / "installments_features.parquet",
    ),
]


def validate_inputs() -> None:
    required = [
        SOURCE / "application_train.parquet",
        SOURCE / "application_test.parquet",
    ]

    required.extend(
        path for _, path in HISTORICAL_FEATURES
    )

    missing = [
        path
        for path in required
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Faltan archivos:\n"
            + "\n".join(str(path) for path in missing)
        )


def build_dataset(
    application_path: Path,
    output_path: Path,
    dataset_name: str,
) -> None:
    print()
    print("=" * 80)
    print(f"CONSOLIDANDO {dataset_name}")
    print("=" * 80)

    connection = DuckDBManager().connect()

    try:
        application = str(application_path.resolve()).replace("'", "''")
        output = str(output_path.resolve()).replace("'", "''")

        joins = []
        selects = ["app.*"]

        aliases = {
            "bureau": "b",
            "bureau_balance": "bb",
            "previous_application": "pa",
            "pos_cash": "pos",
            "credit_card": "cc",
            "installments": "inst",
        }

        flag_names = {
            "bureau": "HAS_BUREAU_HISTORY",
            "bureau_balance": "HAS_BUREAU_BALANCE_HISTORY",
            "previous_application": "HAS_PREVIOUS_APPLICATION_HISTORY",
            "pos_cash": "HAS_POS_HISTORY",
            "credit_card": "HAS_CREDIT_CARD_HISTORY",
            "installments": "HAS_INSTALLMENTS_HISTORY",
        }

        for name, path in HISTORICAL_FEATURES:
            alias = aliases[name]

            escaped_path = str(
                path.resolve()
            ).replace("'", "''")

            joins.append(
                f"""
                LEFT JOIN read_parquet('{escaped_path}') AS {alias}
                    ON app.SK_ID_CURR = {alias}.SK_ID_CURR
                """
            )

            columns = connection.execute(
                f"""
                DESCRIBE
                SELECT *
                FROM read_parquet('{escaped_path}')
                """
            ).fetchall()

            feature_columns = [
                row[0]
                for row in columns
                if row[0] != "SK_ID_CURR"
            ]

            selects.extend(
                f"{alias}.{column}"
                for column in feature_columns
            )

            selects.append(
                f"""
                CASE
                    WHEN {alias}.SK_ID_CURR IS NULL THEN 0
                    ELSE 1
                END AS {flag_names[name]}
                """
            )

        select_sql = ",\n".join(selects)
        join_sql = "\n".join(joins)

        query = f"""
            COPY (
                SELECT
                    {select_sql}

                FROM read_parquet('{application}') AS app

                {join_sql}

                ORDER BY app.SK_ID_CURR
            )
            TO '{output}'
            (
                FORMAT PARQUET,
                COMPRESSION ZSTD
            )
        """

        connection.execute(query)

        row_count = connection.execute(
            f"""
            SELECT COUNT(*)
            FROM read_parquet('{output}')
            """
        ).fetchone()[0]

        column_count = len(
            connection.execute(
                f"""
                DESCRIBE
                SELECT *
                FROM read_parquet('{output}')
                """
            ).fetchall()
        )

        print(f"[OK] {output_path.name}")
        print(f"     Filas:    {row_count:,}")
        print(f"     Columnas: {column_count}")

    finally:
        connection.close()


def main() -> None:
    validate_inputs()

    build_dataset(
        SOURCE / "application_train.parquet",
        OUTPUT / "client_features_train.parquet",
        "TRAIN",
    )

    build_dataset(
        SOURCE / "application_test.parquet",
        OUTPUT / "client_features_test.parquet",
        "TEST",
    )

    print()
    print("=" * 80)
    print("CONSOLIDACION 8.8 FINALIZADA")
    print("=" * 80)


if __name__ == "__main__":
    main()
