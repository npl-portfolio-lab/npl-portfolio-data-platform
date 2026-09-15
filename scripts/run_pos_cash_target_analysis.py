from pathlib import Path

from npl_portfolio.analytics.historical.pos_cash_service import (
    POSCashService,
)

BASE_DIR = Path(__file__).resolve().parents[1]

POS_CASH_PATH = (
    BASE_DIR / "data" / "processed" / "home_credit" / "POS_CASH_balance.parquet"
)

APPLICATION_PATH = (
    BASE_DIR / "data" / "processed" / "home_credit" / "application_train.parquet"
)


def main() -> None:
    service = POSCashService(
        parquet_path=POS_CASH_PATH,
    )

    analysis = service.get_pos_cash_target_analysis(
        application_path=APPLICATION_PATH,
    )

    print()
    print("=" * 90)
    print("HOME CREDIT - POS CASH VS TARGET")
    print("=" * 90)

    print()
    print("COBERTURA DE HISTORIAL")
    print("-" * 90)

    print("Clientes application_train: " f"{analysis['application_clients']:,}")

    print("Con historial POS:          " f"{analysis['clients_with_history']:,}")

    print("Sin historial POS:          " f"{analysis['clients_without_history']:,}")

    print(
        "Cobertura:                   "
        f"{analysis['history_coverage_percentage']:.2f}%"
    )

    print()
    print("FEATURES AGREGADAS")
    print("-" * 90)

    for feature in analysis["aggregated_features"]:
        print(f"  - {feature}")

    print()
    print("POS CASH VS TARGET")
    print("-" * 90)

    for target_result in analysis["target_statistics"]:
        print()
        print(f"TARGET = {target_result['target']}")

        print("Clientes:             " f"{target_result['clients']:,}")

        print("Con historial POS:    " f"{target_result['clients_with_history']:,}")

        print(
            "Cobertura historial:  "
            f"{target_result['history_coverage_percentage']:.2f}%"
        )

        for feature in target_result["features"]:
            print()
            print(f"  {feature['feature']}")

            print("    Registros: " f"{feature['count']:,}")

            print("    Media:     " f"{feature['mean']:,.4f}")

            print("    Mediana:   " f"{feature['median']:,.4f}")

            print("    P25:       " f"{feature['p25']:,.4f}")

            print("    P75:       " f"{feature['p75']:,.4f}")

            print("    Mínimo:    " f"{feature['min']:,.4f}")

            print("    Máximo:    " f"{feature['max']:,.4f}")


if __name__ == "__main__":
    main()
