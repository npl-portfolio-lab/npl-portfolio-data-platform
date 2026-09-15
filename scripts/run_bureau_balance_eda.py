from pathlib import Path

from npl_portfolio.analytics.historical.bureau_balance_service import (
    BureauBalanceService,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BUREAU_BALANCE_PATH = (
    PROJECT_ROOT / "data" / "processed" / "home_credit" / "bureau_balance.parquet"
)

BUREAU_PATH = PROJECT_ROOT / "data" / "processed" / "home_credit" / "bureau.parquet"


def main() -> None:
    service = BureauBalanceService(
        parquet_path=BUREAU_BALANCE_PATH,
    )

    print("=" * 80)
    print("EDA HISTORICO - BUREAU BALANCE")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Dataset
    # ------------------------------------------------------------------
    overview = service.get_dataset_overview()

    print("\n1. DATASET")
    print("-" * 80)
    print(f"Registros: {overview['rows']:,}")
    print(f"Columnas: {overview['columns']:,}")

    # ------------------------------------------------------------------
    # 2. Créditos Bureau
    # ------------------------------------------------------------------
    clients = service.get_unique_clients(
        client_column="SK_ID_BUREAU",
    )

    print("\n2. CREDITOS BUREAU")
    print("-" * 80)
    print("SK_ID_BUREAU unicos: " f"{clients['unique_clients']:,}")

    # ------------------------------------------------------------------
    # 3. Registros mensuales por crédito
    # ------------------------------------------------------------------
    records = service.get_records_per_client_summary(
        client_column="SK_ID_BUREAU",
    )

    print("\n3. REGISTROS MENSUALES POR SK_ID_BUREAU")
    print("-" * 80)

    for key, value in records.items():
        print(f"{key}: {value}")

    # ------------------------------------------------------------------
    # 4. STATUS
    # ------------------------------------------------------------------
    categorical = service.get_categorical_summary(
        columns=["STATUS"],
        top_n=20,
    )

    print("\n4. DISTRIBUCION DE STATUS")
    print("-" * 80)

    for column_summary in categorical["statistics"]:
        print(f"\nColumna: {column_summary['column']}")
        print("Valores unicos: " f"{column_summary['unique_values']}")
        print(
            "Nulos: "
            f"{column_summary['null_count']:,} "
            f"({column_summary['null_percentage']}%)"
        )

        print("\nValores:")

        for item in column_summary["top_categories"]:
            print(
                f"  {item['value']}: " f"{item['count']:,} " f"({item['percentage']}%)"
            )

        # ------------------------------------------------------------------
    # 5. MONTHS_BALANCE
    # ------------------------------------------------------------------
    numeric = service.get_numeric_summary(
        columns=["MONTHS_BALANCE"],
    )

    print("\n5. MONTHS_BALANCE")
    print("-" * 80)

    for summary in numeric["statistics"]:
        print(f"\nColumna: {summary['column']}")
        print(f"Validos: {summary['count']:,}")
        print(f"Nulos: {summary['null_count']:,}")
        print("Porcentaje nulos: " f"{summary['null_percentage']}%")
        print(f"Media: {summary['mean']}")
        print(f"Mediana: {summary['median']}")
        print(f"P25: {summary['p25']}")
        print(f"P75: {summary['p75']}")
        print(f"Minimo: {summary['min']}")
        print(f"Maximo: {summary['max']}")

    # ------------------------------------------------------------------
    # 6. Diagnóstico de mapeo contra bureau
    # ------------------------------------------------------------------
    mapping = service.get_mapping_diagnostics(
        bureau_path=BUREAU_PATH,
    )

    print("\n6. MAPEO BUREAU_BALANCE -> BUREAU")
    print("-" * 80)

    print("Registros totales: " f"{mapping['total_rows']:,}")
    print(
        "Registros mapeados: "
        f"{mapping['mapped_rows']:,} "
        f"({mapping['mapped_percentage']}%)"
    )
    print(
        "Registros no mapeados: "
        f"{mapping['unmapped_rows']:,} "
        f"({mapping['unmapped_percentage']}%)"
    )
    print("SK_ID_BUREAU no mapeables: " f"{mapping['unmapped_bureau_ids']:,}")

    print("\n" + "=" * 80)
    print("EDA BUREAU BALANCE FINALIZADO")
    print("=" * 80)


if __name__ == "__main__":
    main()
