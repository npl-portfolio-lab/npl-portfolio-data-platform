from pathlib import Path

from npl_portfolio.analytics.eda_service import (
    EDAService,
)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    parquet_path = (
        project_root
        / "data"
        / "processed"
        / "home_credit"
        / "application_train.parquet"
    )

    service = EDAService(parquet_path=parquet_path)

    overview = service.get_dataset_overview()

    target = service.get_target_distribution()

    missing = service.get_missing_values_summary(top_n=20)

    numeric = service.get_numeric_summary(
        columns=[
            "AMT_INCOME_TOTAL",
            "AMT_CREDIT",
            "AMT_ANNUITY",
            "AMT_GOODS_PRICE",
            "DAYS_BIRTH",
            "DAYS_EMPLOYED",
        ]
    )

    employed_sentinel = service.get_value_frequency(
        column="DAYS_EMPLOYED",
        value=365243,
    )

    income_percentiles = service.get_percentiles(
        column="AMT_INCOME_TOTAL",
        percentiles=[
            0.90,
            0.95,
            0.99,
            0.995,
            0.999,
        ],
    )

    categorical = service.get_categorical_summary(
        columns=[
            "NAME_CONTRACT_TYPE",
            "CODE_GENDER",
            "FLAG_OWN_CAR",
            "FLAG_OWN_REALTY",
            "NAME_INCOME_TYPE",
            "NAME_EDUCATION_TYPE",
            "NAME_FAMILY_STATUS",
            "NAME_HOUSING_TYPE",
            "OCCUPATION_TYPE",
            "ORGANIZATION_TYPE",
        ],
        top_n=10,
    )

    categorical_target = service.get_target_rate_by_category(
        columns=[
            "NAME_CONTRACT_TYPE",
            "CODE_GENDER",
            "FLAG_OWN_CAR",
            "FLAG_OWN_REALTY",
            "NAME_INCOME_TYPE",
            "NAME_EDUCATION_TYPE",
            "NAME_FAMILY_STATUS",
            "NAME_HOUSING_TYPE",
            "OCCUPATION_TYPE",
            "ORGANIZATION_TYPE",
        ],
        top_n=20,
    )

    numeric_target = service.get_numeric_summary_by_target(
        columns=[
            "AMT_INCOME_TOTAL",
            "AMT_CREDIT",
            "AMT_ANNUITY",
            "AMT_GOODS_PRICE",
            "DAYS_BIRTH",
            "DAYS_EMPLOYED",
        ]
    )

    print()
    print("HOME CREDIT - EDA")
    print("=" * 90)

    print()
    print("DATASET")
    print("-" * 90)

    print(f"Archivo:       " f"{overview['file']}")

    print(f"Filas:         " f"{overview['rows']:,}")

    print(f"Columnas:      " f"{overview['columns']:,}")

    print(f"Row groups:    " f"{overview['row_groups']:,}")

    print()
    print("VARIABLE OBJETIVO")
    print("-" * 90)

    print(f"Clientes:      " f"{target['total_clients']:,}")

    print(
        f"TARGET = 0:    "
        f"{target['target_0']:,} "
        f"({target['target_0_percentage']:.2f}%)"
    )

    print(
        f"TARGET = 1:    "
        f"{target['target_1']:,} "
        f"({target['target_1_percentage']:.2f}%)"
    )

    print()
    print("VALORES FALTANTES")
    print("-" * 90)

    print(f"Columnas totales:       " f"{missing['total_columns']:,}")

    print(f"Con valores faltantes:  " f"{missing['columns_with_missing']:,}")

    print(f"Sin valores faltantes:  " f"{missing['columns_without_missing']:,}")

    print(f"Con >= 30% faltantes:   " f"{missing['columns_over_30_percent']:,}")

    print(f"Con >= 50% faltantes:   " f"{missing['columns_over_50_percent']:,}")

    print()
    print("TOP 20 COLUMNAS CON MÁS NULOS")
    print("-" * 90)

    for index, item in enumerate(
        missing["top_missing_columns"],
        start=1,
    ):
        print(
            f"{index:>2}. "
            f"{item['column']:<35} "
            f"{item['null_count']:>10,} "
            f"({item['null_percentage']:>6.2f}%)"
        )

    print()
    print("VARIABLES NUMÉRICAS")
    print("-" * 90)

    for item in numeric["statistics"]:
        print()
        print(f"{item['column']}")

        print(f"  Registros válidos: " f"{item['count']:,}")

        print(
            f"  Nulos:             "
            f"{item['null_count']:,} "
            f"({item['null_percentage']:.2f}%)"
        )

        print(f"  Media:             " f"{item['mean']:,.2f}")

        print(f"  Mediana:           " f"{item['median']:,.2f}")

        print(f"  Desviación:        " f"{item['std']:,.2f}")

        print(f"  Mínimo:            " f"{item['min']:,.2f}")

        print(f"  Percentil 25:      " f"{item['p25']:,.2f}")

        print(f"  Percentil 75:      " f"{item['p75']:,.2f}")

        print(f"  Máximo:            " f"{item['max']:,.2f}")

    print()
    print("VALORES ESPECIALES Y OUTLIERS")
    print("-" * 90)

    print()
    print("DAYS_EMPLOYED")

    print(f"  Valor especial:    " f"{employed_sentinel['value']:,}")

    print(f"  Registros:         " f"{employed_sentinel['occurrences']:,}")

    print(f"  Porcentaje:        " f"{employed_sentinel['percentage']:.2f}%")

    print()
    print("AMT_INCOME_TOTAL")

    for percentile, value in income_percentiles["percentiles"].items():
        print(f"  {percentile:<8} " f"{value:>15,.2f}")

    print()
    print("VARIABLES CATEGÓRICAS")
    print("-" * 90)

    for item in categorical["statistics"]:
        print()
        print(f"{item['column']}")

        print(f"  Valores únicos:     " f"{item['unique_values']:,}")

        print(
            f"  Nulos:              "
            f"{item['null_count']:,} "
            f"({item['null_percentage']:.2f}%)"
        )

        for category in item["top_categories"]:
            print(
                f"    {category['value']:<30} "
                f"{category['count']:>10,} "
                f"({category['percentage']:>6.2f}%)"
            )

    print()
    print("TARGET VS VARIABLES CATEGÓRICAS")
    print("-" * 90)

    for item in categorical_target["statistics"]:
        print()
        print(f"{item['column']}")

        for category in item["categories"]:
            print(
                f"  {category['value']:<30} "
                f"Clientes: "
                f"{category['clients']:>8,} | "
                f"TARGET=1: "
                f"{category['target_1']:>7,} | "
                f"Tasa: "
                f"{category['target_rate']:>6.2f}%"
            )

    print()
    print("TARGET VS VARIABLES NUMÉRICAS")
    print("-" * 90)

    for item in numeric_target["statistics"]:
        print()
        print(f"{item['column']}")

        for target_stats in item["target_statistics"]:
            print(f"  TARGET = " f"{target_stats['target']}")

            print(f"    Registros:   " f"{target_stats['count']:,}")

            print(f"    Media:       " f"{target_stats['mean']:,.2f}")

            print(f"    Mediana:     " f"{target_stats['median']:,.2f}")

            print(f"    P25:         " f"{target_stats['p25']:,.2f}")

            print(f"    P75:         " f"{target_stats['p75']:,.2f}")

            print(f"    Mínimo:      " f"{target_stats['min']:,.2f}")

            print(f"    Máximo:      " f"{target_stats['max']:,.2f}")

    print()
    print("=" * 90)


if __name__ == "__main__":
    main()
