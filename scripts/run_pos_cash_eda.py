from pathlib import Path

from npl_portfolio.analytics.historical.pos_cash_service import (
    POSCashService,
)

BASE_DIR = Path(__file__).resolve().parents[1]

POS_CASH_PATH = (
    BASE_DIR / "data" / "processed" / "home_credit" / "POS_CASH_balance.parquet"
)


def main() -> None:
    service = POSCashService(
        parquet_path=POS_CASH_PATH,
    )

    overview = service.get_dataset_overview()

    clients = service.get_unique_clients()

    records_per_client = service.get_records_per_client_summary()

    categorical = service.get_categorical_summary(
        columns=[
            "NAME_CONTRACT_STATUS",
        ],
        top_n=10,
    )

    numeric = service.get_numeric_summary(
        columns=[
            "MONTHS_BALANCE",
            "CNT_INSTALMENT",
            "CNT_INSTALMENT_FUTURE",
            "SK_DPD",
            "SK_DPD_DEF",
        ]
    )

    print()
    print("=" * 90)
    print("HOME CREDIT - POS CASH BALANCE EDA")
    print("=" * 90)

    print()
    print("DATASET")
    print("-" * 90)
    print(f"Archivo:          {overview['file']}")
    print(f"Filas:            {overview['rows']:,}")
    print(f"Columnas:         {overview['columns']:,}")

    print()
    print("CLIENTES")
    print("-" * 90)
    print("Registros históricos: " f"{clients['total_rows']:,}")
    print("Clientes únicos:       " f"{clients['unique_clients']:,}")
    print("Promedio por cliente:  " f"{clients['average_records_per_client']:.2f}")

    print()
    print("REGISTROS POR CLIENTE")
    print("-" * 90)
    print(f"Clientes: {records_per_client['clients']:,}")
    print(f"Media:    {records_per_client['mean_records']:.2f}")
    print(f"Mediana:  {records_per_client['median_records']:.2f}")
    print(f"Mínimo:   {records_per_client['min_records']:,}")
    print(f"P25:      {records_per_client['p25_records']:.2f}")
    print(f"P75:      {records_per_client['p75_records']:.2f}")
    print(f"P90:      {records_per_client['p90_records']:.2f}")
    print(f"P95:      {records_per_client['p95_records']:.2f}")
    print(f"P99:      {records_per_client['p99_records']:.2f}")
    print(f"Máximo:   {records_per_client['max_records']:,}")

    print()
    print("VARIABLES CATEGÓRICAS")
    print("-" * 90)

    for statistic in categorical["statistics"]:
        print()
        print(statistic["column"])
        print("  Valores únicos: " f"{statistic['unique_values']:,}")
        print(
            "  Nulos:          "
            f"{statistic['null_count']:,} "
            f"({statistic['null_percentage']:.2f}%)"
        )

        for category in statistic["top_categories"]:
            print(
                f"    {category['value']:<35} "
                f"{category['count']:>12,} "
                f"({category['percentage']:>6.2f}%)"
            )

    print()
    print("VARIABLES NUMÉRICAS")
    print("-" * 90)

    for statistic in numeric["statistics"]:
        print()
        print(statistic["column"])

        print("  Registros válidos: " f"{statistic['count']:,}")

        print(
            "  Nulos:             "
            f"{statistic['null_count']:,} "
            f"({statistic['null_percentage']:.2f}%)"
        )

        values = [
            ("Media", statistic["mean"]),
            ("Mediana", statistic["median"]),
            ("P25", statistic["p25"]),
            ("P75", statistic["p75"]),
            ("Mínimo", statistic["min"]),
            ("Máximo", statistic["max"]),
        ]

        for label, value in values:
            if value is None:
                formatted_value = "N/A"
            else:
                formatted_value = f"{value:,.2f}"

            print(f"  {label + ':':<19} " f"{formatted_value}")

    print()
    print("=" * 90)


if __name__ == "__main__":
    main()
