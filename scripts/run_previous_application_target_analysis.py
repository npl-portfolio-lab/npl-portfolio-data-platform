from pathlib import Path

from npl_portfolio.analytics.historical_eda_service import (
    HistoricalEDAService,
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

    previous_path = (
        data_path
        / "previous_application.parquet"
    )

    application_path = (
        data_path
        / "application_train.parquet"
    )

    service = HistoricalEDAService(
        parquet_path=previous_path
    )

    analysis = (
        service.get_previous_application_target_analysis(
            application_path=application_path
        )
    )

    print()
    print(
        "HOME CREDIT - PREVIOUS APPLICATION VS TARGET"
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
        f"Con solicitudes previas:    "
        f"{analysis['clients_with_history']:,}"
    )

    print(
        f"Sin solicitudes previas:    "
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
    print("PREVIOUS APPLICATION VS TARGET")
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
            f"Con historial:         "
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
