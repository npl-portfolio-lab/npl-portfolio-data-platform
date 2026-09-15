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

    data_path = (
        project_root
        / "data"
        / "processed"
        / "home_credit"
    )

    pos_path = (
        data_path
        / "POS_CASH_balance.parquet"
    )

    application_path = (
        data_path
        / "application_train.parquet"
    )

    service = DuckDBHistoricalService(
        parquet_path=pos_path
    )

    analysis = (
        service.get_pos_cash_target_analysis(
            application_path=application_path
        )
    )

    print()
    print(
        "HOME CREDIT - POS CASH VS TARGET"
    )
    print("=" * 90)

    print()
    print("COBERTURA DE HISTORIAL")
    print("-" * 90)

    print(
        f"Clientes application_train: "
        f"{analysis['application_clients']:,}"
    )

    print(
        f"Con historial POS:          "
        f"{analysis['clients_with_history']:,}"
    )

    print(
        f"Sin historial POS:          "
        f"{analysis['clients_without_history']:,}"
    )

    print(
        f"Cobertura:                   "
        f"{analysis['history_coverage_percentage']:.2f}%"
    )

    print()
    print("FEATURES AGREGADAS")
    print("-" * 90)

    for feature in analysis[
        "aggregated_features"
    ]:
        print(
            f"  - {feature}"
        )

    print()
    print("POS CASH VS TARGET")
    print("-" * 90)

    for target_data in analysis[
        "target_statistics"
    ]:
        print()
        print(
            f"TARGET = "
            f"{target_data['target']}"
        )

        print(
            f"Clientes:             "
            f"{target_data['clients']:,}"
        )

        print(
            f"Con historial POS:     "
            f"{target_data['clients_with_history']:,}"
        )

        print(
            f"Cobertura historial:   "
            f"{target_data['history_coverage_percentage']:.2f}%"
        )

        print()

        for feature in target_data[
            "features"
        ]:
            print(
                f"  {feature['feature']}"
            )

            print(
                f"    Registros: "
                f"{feature['count']:,}"
            )

            print(
                f"    Media:     "
                f"{feature['mean']:,.4f}"
            )

            print(
                f"    Mediana:   "
                f"{feature['median']:,.4f}"
            )

            print(
                f"    P25:       "
                f"{feature['p25']:,.4f}"
            )

            print(
                f"    P75:       "
                f"{feature['p75']:,.4f}"
            )

            print(
                f"    Mínimo:    "
                f"{feature['min']:,.4f}"
            )

            print(
                f"    Máximo:    "
                f"{feature['max']:,.4f}"
            )

            print()

    print("=" * 90)


if __name__ == "__main__":
    main()
