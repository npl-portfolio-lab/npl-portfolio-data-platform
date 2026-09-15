from pathlib import Path

from npl_portfolio.analytics.duckdb_historical_service import (
    DuckDBHistoricalService,
)


def main() -> None:
    project_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    parquet_path = (
        project_root
        / "data"
        / "processed"
        / "home_credit"
        / "POS_CASH_balance.parquet"
    )

    service = DuckDBHistoricalService(
        parquet_path=parquet_path
    )

    overview = (
        service.get_dataset_overview()
    )

    clients = (
        service.get_unique_clients()
    )

    records_per_client = (
        service.get_records_per_client_summary()
    )

    categorical = (
        service.get_categorical_summary(
            columns=[
                "NAME_CONTRACT_STATUS",
            ],
            top_n=15,
        )
    )

    numeric = (
        service.get_numeric_summary(
            columns=[
                "MONTHS_BALANCE",
                "CNT_INSTALMENT",
                "CNT_INSTALMENT_FUTURE",
                "SK_DPD",
                "SK_DPD_DEF",
            ]
        )
    )

    print()
    print(
        "HOME CREDIT - POS CASH BALANCE EDA"
    )
    print("=" * 90)

    print()
    print("DATASET")
    print("-" * 90)

    print(
        f"Archivo:          "
        f"{overview['file']}"
    )

    print(
        f"Filas:            "
        f"{overview['rows']:,}"
    )

    print(
        f"Columnas:         "
        f"{overview['columns']:,}"
    )

    print()
    print("CLIENTES")
    print("-" * 90)

    print(
        f"Registros históricos: "
        f"{clients['total_rows']:,}"
    )

    print(
        f"Clientes únicos:       "
        f"{clients['unique_clients']:,}"
    )

    print(
        f"Promedio por cliente:  "
        f"{clients['average_records_per_client']:,.2f}"
    )

    print()
    print("REGISTROS POR CLIENTE")
    print("-" * 90)

    print(
        f"Clientes: "
        f"{records_per_client['clients']:,}"
    )

    print(
        f"Media:    "
        f"{records_per_client['mean_records']:,.2f}"
    )

    print(
        f"Mediana:  "
        f"{records_per_client['median_records']:,.2f}"
    )

    print(
        f"Mínimo:   "
        f"{records_per_client['min_records']:,}"
    )

    print(
        f"P25:      "
        f"{records_per_client['p25_records']:,.2f}"
    )

    print(
        f"P75:      "
        f"{records_per_client['p75_records']:,.2f}"
    )

    print(
        f"P90:      "
        f"{records_per_client['p90_records']:,.2f}"
    )

    print(
        f"P95:      "
        f"{records_per_client['p95_records']:,.2f}"
    )

    print(
        f"P99:      "
        f"{records_per_client['p99_records']:,.2f}"
    )

    print(
        f"Máximo:   "
        f"{records_per_client['max_records']:,}"
    )

    print()
    print("VARIABLES CATEGÓRICAS")
    print("-" * 90)

    for item in categorical[
        "statistics"
    ]:
        print()
        print(
            item["column"]
        )

        print(
            f"  Valores únicos: "
            f"{item['unique_values']:,}"
        )

        print(
            f"  Nulos:          "
            f"{item['null_count']:,} "
            f"({item['null_percentage']:.2f}%)"
        )

        for category in item[
            "top_categories"
        ]:
            print(
                f"    "
                f"{category['value']:<35} "
                f"{category['count']:>12,} "
                f"({category['percentage']:>6.2f}%)"
            )

    print()
    print("VARIABLES NUMÉRICAS")
    print("-" * 90)

    for item in numeric[
        "statistics"
    ]:
        print()
        print(
            item["column"]
        )

        print(
            f"  Registros válidos: "
            f"{item['count']:,}"
        )

        print(
            f"  Nulos:             "
            f"{item['null_count']:,} "
            f"({item['null_percentage']:.2f}%)"
        )

        print(
            f"  Media:             "
            f"{item['mean']:,.2f}"
        )

        print(
            f"  Mediana:           "
            f"{item['median']:,.2f}"
        )

        print(
            f"  P25:               "
            f"{item['p25']:,.2f}"
        )

        print(
            f"  P75:               "
            f"{item['p75']:,.2f}"
        )

        print(
            f"  Mínimo:            "
            f"{item['min']:,.2f}"
        )

        print(
            f"  Máximo:            "
            f"{item['max']:,.2f}"
        )

    print()
    print("=" * 90)


if __name__ == "__main__":
    main()
