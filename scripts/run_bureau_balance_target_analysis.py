from pathlib import Path

from npl_portfolio.analytics.historical.bureau_balance_service import (
    BureauBalanceService,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "home_credit"
)

BUREAU_BALANCE_PATH = DATA_PATH / "bureau_balance.parquet"
BUREAU_PATH = DATA_PATH / "bureau.parquet"
APPLICATION_PATH = DATA_PATH / "application_train.parquet"


def print_feature_statistics(
    target_statistics: list[dict],
) -> None:
    for target_result in target_statistics:
        target = target_result["target"]

        print("\n" + "=" * 80)
        print(f"TARGET = {target}")
        print("=" * 80)

        print(
            "Clientes: "
            f"{target_result['clients']:,}"
        )

        print(
            "Clientes con historial: "
            f"{target_result['clients_with_history']:,}"
        )

        print(
            "Cobertura: "
            f"{target_result['history_coverage_percentage']}%"
        )

        print("\nMETRICAS")
        print("-" * 80)

        for feature in target_result["features"]:
            print(f"\n{feature['feature']}")
            print(f"  count:   {feature['count']:,}")
            print(f"  mean:    {feature['mean']}")
            print(f"  median:  {feature['median']}")
            print(f"  p25:     {feature['p25']}")
            print(f"  p75:     {feature['p75']}")
            print(f"  min:     {feature['min']}")
            print(f"  max:     {feature['max']}")


def main() -> None:
    service = BureauBalanceService(
        parquet_path=BUREAU_BALANCE_PATH,
    )

    print("=" * 80)
    print("BUREAU BALANCE - ANALISIS CONTRA TARGET")
    print("=" * 80)

    result = service.get_bureau_balance_target_analysis(
        bureau_path=BUREAU_PATH,
        application_path=APPLICATION_PATH,
    )

    # ------------------------------------------------------------------
    # 1. Cobertura sobre application_train
    # ------------------------------------------------------------------
    print("\n1. COBERTURA")
    print("-" * 80)

    print(
        "Clientes application_train: "
        f"{result['application_clients']:,}"
    )

    print(
        "Clientes con historial bureau_balance: "
        f"{result['clients_with_history']:,}"
    )

    print(
        "Clientes sin historial bureau_balance: "
        f"{result['clients_without_history']:,}"
    )

    print(
        "Cobertura total: "
        f"{result['history_coverage_percentage']}%"
    )

    # ------------------------------------------------------------------
    # 2. Diagnóstico de mapeo
    # ------------------------------------------------------------------
    mapping = result["mapping_diagnostics"]

    print("\n2. MAPEO BUREAU_BALANCE -> BUREAU")
    print("-" * 80)

    print(
        "Registros totales: "
        f"{mapping['total_rows']:,}"
    )

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

    print(
        "SK_ID_BUREAU no mapeables: "
        f"{mapping['unmapped_bureau_ids']:,}"
    )

    # ------------------------------------------------------------------
    # 3. Features agregadas
    # ------------------------------------------------------------------
    print("\n3. FEATURES AGREGADAS")
    print("-" * 80)

    print(
        "Cantidad de features: "
        f"{len(result['aggregated_features'])}"
    )

    for feature in result["aggregated_features"]:
        print(f"  - {feature}")

    # ------------------------------------------------------------------
    # 4. Comparación TARGET
    # ------------------------------------------------------------------
    print("\n4. COMPARACION POR TARGET")

    print_feature_statistics(
        result["target_statistics"],
    )

    print("\n" + "=" * 80)
    print("ANALISIS BUREAU BALANCE FINALIZADO")
    print("=" * 80)


if __name__ == "__main__":
    main()
