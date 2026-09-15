from pathlib import Path

from npl_portfolio.analytics.historical.installments_service import (
    InstallmentsService,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INSTALLMENTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "home_credit"
    / "installments_payments.parquet"
)

APPLICATION_PATH = (
    PROJECT_ROOT / "data" / "processed" / "home_credit" / "application_train.parquet"
)


def print_feature_statistics(
    feature: dict,
) -> None:
    print(f"\n  {feature['feature']}")
    print(f"    Registros: {feature['count']:,}")
    print(f"    Media:     {feature['mean']:,.4f}")
    print(f"    Mediana:   {feature['median']:,.4f}")
    print(f"    P25:       {feature['p25']:,.4f}")
    print(f"    P75:       {feature['p75']:,.4f}")
    print(f"    Mínimo:    {feature['min']:,.4f}")
    print(f"    Máximo:    {feature['max']:,.4f}")


def main() -> None:
    service = InstallmentsService(INSTALLMENTS_PATH)

    result = service.get_installments_target_analysis(
        application_path=APPLICATION_PATH,
    )

    print("\nHOME CREDIT - INSTALLMENTS PAYMENTS VS TARGET")
    print("=" * 90)

    print("\nCOBERTURA DE HISTORIAL")
    print("-" * 90)

    print("Clientes application_train: " f"{result['application_clients']:,}")

    print("Con historial installments:  " f"{result['clients_with_history']:,}")

    print("Sin historial installments:  " f"{result['clients_without_history']:,}")

    print(
        "Cobertura:                   " f"{result['history_coverage_percentage']:.2f}%"
    )

    print("\nFEATURES AGREGADAS")
    print("-" * 90)

    for feature in result["aggregated_features"]:
        print(f"  - {feature}")

    print("\nINSTALLMENTS PAYMENTS VS TARGET")
    print("-" * 90)

    for target_result in result["target_statistics"]:
        print(f"\nTARGET = {target_result['target']}")

        print("Clientes:                    " f"{target_result['clients']:,}")

        print(
            "Con historial installments: " f"{target_result['clients_with_history']:,}"
        )

        print(
            "Cobertura historial:         "
            f"{target_result['history_coverage_percentage']:.2f}%"
        )

        for feature in target_result["features"]:
            print_feature_statistics(feature)

    print("\n" + "=" * 90)


if __name__ == "__main__":
    main()
