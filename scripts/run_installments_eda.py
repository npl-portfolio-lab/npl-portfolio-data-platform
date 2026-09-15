from pathlib import Path

from npl_portfolio.analytics.historical.installments_service import (
    InstallmentsService,
)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    parquet_path = (
        project_root
        / "data"
        / "processed"
        / "home_credit"
        / "installments_payments.parquet"
    )

    service = InstallmentsService(parquet_path=parquet_path)

    overview = service.get_dataset_overview()

    clients = service.get_unique_clients()

    records_per_client = service.get_records_per_client_summary()

    numeric = service.get_numeric_summary(
        columns=[
            "NUM_INSTALMENT_VERSION",
            "NUM_INSTALMENT_NUMBER",
            "DAYS_INSTALMENT",
            "DAYS_ENTRY_PAYMENT",
            "AMT_INSTALMENT",
            "AMT_PAYMENT",
        ]
    )

    print()
    print("HOME CREDIT - INSTALLMENTS PAYMENTS EDA")
    print("=" * 90)

    print()
    print("DATASET")
    print("-" * 90)

    print(f"Archivo:          " f"{overview['file']}")

    print(f"Filas:            " f"{overview['rows']:,}")

    print(f"Columnas:         " f"{overview['columns']:,}")

    print()
    print("CLIENTES")
    print("-" * 90)

    print(f"Registros históricos: " f"{clients['total_rows']:,}")

    print(f"Clientes únicos:       " f"{clients['unique_clients']:,}")

    print(f"Promedio por cliente:  " f"{clients['average_records_per_client']:,.2f}")

    print()
    print("REGISTROS POR CLIENTE")
    print("-" * 90)

    print(f"Clientes: " f"{records_per_client['clients']:,}")

    print(f"Media:    " f"{records_per_client['mean_records']:,.2f}")

    print(f"Mediana:  " f"{records_per_client['median_records']:,.2f}")

    print(f"Mínimo:   " f"{records_per_client['min_records']:,}")

    print(f"P25:      " f"{records_per_client['p25_records']:,.2f}")

    print(f"P75:      " f"{records_per_client['p75_records']:,.2f}")

    print(f"P90:      " f"{records_per_client['p90_records']:,.2f}")

    print(f"P95:      " f"{records_per_client['p95_records']:,.2f}")

    print(f"P99:      " f"{records_per_client['p99_records']:,.2f}")

    print(f"Máximo:   " f"{records_per_client['max_records']:,}")

    print()
    print("VARIABLES NUMÉRICAS")
    print("-" * 90)

    for item in numeric["statistics"]:
        print()
        print(item["column"])

        print(f"  Registros válidos: " f"{item['count']:,}")

        print(
            f"  Nulos:             "
            f"{item['null_count']:,} "
            f"({item['null_percentage']:.2f}%)"
        )

        print(f"  Media:             " f"{item['mean']:,.2f}")

        print(f"  Mediana:           " f"{item['median']:,.2f}")

        print(f"  P25:               " f"{item['p25']:,.2f}")

        print(f"  P75:               " f"{item['p75']:,.2f}")

        print(f"  Mínimo:            " f"{item['min']:,.2f}")

        print(f"  Máximo:            " f"{item['max']:,.2f}")

    print()
    print("=" * 90)


if __name__ == "__main__":
    main()
