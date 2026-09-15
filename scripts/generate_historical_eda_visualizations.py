from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from npl_portfolio.analytics.historical.bureau_balance_service import (
    BureauBalanceService,
)
from npl_portfolio.analytics.historical.credit_card_service import (
    CreditCardService,
)
from npl_portfolio.analytics.historical.installments_service import (
    InstallmentsService,
)
from npl_portfolio.analytics.historical.pos_cash_service import (
    POSCashService,
)
from npl_portfolio.analytics.historical_eda_service import (
    HistoricalEDAService,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "home_credit"

OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "eda" / "historical"

APPLICATION_PATH = DATA_DIR / "application_train.parquet"
BUREAU_PATH = DATA_DIR / "bureau.parquet"
BUREAU_BALANCE_PATH = DATA_DIR / "bureau_balance.parquet"
PREVIOUS_PATH = DATA_DIR / "previous_application.parquet"
POS_PATH = DATA_DIR / "POS_CASH_balance.parquet"
CREDIT_CARD_PATH = DATA_DIR / "credit_card_balance.parquet"
INSTALLMENTS_PATH = DATA_DIR / "installments_payments.parquet"


def save_figure(filename: str) -> None:
    output_path = OUTPUT_DIR / filename

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    print(f"[OK] {output_path}")


def get_feature_statistic(
    analysis: dict[str, Any],
    feature_name: str,
    statistic: str = "mean",
) -> dict[int, float]:
    """
    Extrae una estadística de una feature para cada TARGET.
    """

    values: dict[int, float] = {}

    for target_data in analysis["target_statistics"]:
        target = int(target_data["target"])

        feature = next(
            (
                item
                for item in target_data["features"]
                if item["feature"] == feature_name
            ),
            None,
        )

        if feature is None:
            raise ValueError(f"No se encontró la feature {feature_name}")

        value = feature.get(statistic)

        if value is None:
            raise ValueError(f"No existe {statistic} para {feature_name}")

        values[target] = float(value)

    return values


def plot_coverage(
    analyses: dict[str, dict[str, Any]],
) -> None:
    """
    Compara la cobertura de las fuentes históricas.
    """

    sources = list(analyses.keys())

    coverage = []

    for source in sources:
        analysis = analyses[source]

        if source == "Bureau":
            value = analysis["bureau_coverage_percentage"]
        else:
            value = analysis["history_coverage_percentage"]

        coverage.append(float(value))

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(
        sources,
        coverage,
    )

    ax.set_title("Cobertura de información histórica por fuente")
    ax.set_ylabel("Clientes con historial (%)")
    ax.set_ylim(0, 100)

    ax.tick_params(
        axis="x",
        rotation=25,
    )

    for bar, value in zip(
        bars,
        coverage,
        strict=True,
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}%",
            ha="center",
            va="bottom",
        )

    save_figure("01_history_coverage.png")


def plot_feature_comparison(
    analysis: dict[str, Any],
    features: list[str],
    labels: list[str],
    title: str,
    filename: str,
    multiplier: float = 1.0,
) -> None:
    """
    Compara medias de varias features entre TARGET 0 y TARGET 1.
    """

    target_0_values = []
    target_1_values = []

    for feature in features:
        values = get_feature_statistic(
            analysis=analysis,
            feature_name=feature,
            statistic="mean",
        )

        target_0_values.append(values.get(0, 0.0) * multiplier)

        target_1_values.append(values.get(1, 0.0) * multiplier)

    positions = list(range(len(features)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    left_positions = [position - width / 2 for position in positions]

    right_positions = [position + width / 2 for position in positions]

    bars_0 = ax.bar(
        left_positions,
        target_0_values,
        width,
        label="TARGET = 0",
    )

    bars_1 = ax.bar(
        right_positions,
        target_1_values,
        width,
        label="TARGET = 1",
    )

    ax.set_title(title)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.legend()

    for bars in (bars_0, bars_1):
        for bar in bars:
            value = bar.get_height()

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    save_figure(filename)


def print_available_features(
    analyses: dict[str, dict[str, Any]],
) -> None:
    """
    Muestra las features disponibles para trazabilidad.
    """

    print("\nFEATURES DISPONIBLES")
    print("-" * 80)

    for source, analysis in analyses.items():
        print(f"\n{source}:")

        for feature in analysis["aggregated_features"]:
            print(f"  - {feature}")


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("GENERANDO VISUALIZACIONES EDA - HISTORICOS")
    print("=" * 80)

    bureau_service = HistoricalEDAService(
        parquet_path=BUREAU_PATH,
    )

    previous_service = HistoricalEDAService(
        parquet_path=PREVIOUS_PATH,
    )

    pos_service = POSCashService(
        parquet_path=POS_PATH,
    )

    credit_card_service = CreditCardService(
        parquet_path=CREDIT_CARD_PATH,
    )

    installments_service = InstallmentsService(
        parquet_path=INSTALLMENTS_PATH,
    )

    bureau_balance_service = BureauBalanceService(
        parquet_path=BUREAU_BALANCE_PATH,
    )

    print("\n[1/6] Analizando Bureau...")

    bureau = bureau_service.get_bureau_target_analysis(
        application_path=APPLICATION_PATH,
    )

    print("[2/6] Analizando Previous Application...")

    previous = previous_service.get_previous_application_target_analysis(
        application_path=APPLICATION_PATH,
    )

    print("[3/6] Analizando POS CASH...")

    pos = pos_service.get_pos_cash_target_analysis(
        application_path=APPLICATION_PATH,
    )

    print("[4/6] Analizando Credit Card...")

    credit_card = credit_card_service.get_credit_card_target_analysis(
        application_path=APPLICATION_PATH,
    )

    print("[5/6] Analizando Installments...")

    installments = installments_service.get_installments_target_analysis(
        application_path=APPLICATION_PATH,
    )

    print("[6/6] Analizando Bureau Balance...")

    bureau_balance = bureau_balance_service.get_bureau_balance_target_analysis(
        bureau_path=BUREAU_PATH,
        application_path=APPLICATION_PATH,
    )

    analyses = {
        "Bureau": bureau,
        "Previous": previous,
        "POS": pos,
        "Credit Card": credit_card,
        "Installments": installments,
        "Bureau Balance": bureau_balance,
    }

    print_available_features(analyses)

    plot_coverage(analyses)

    plot_feature_comparison(
        analysis=bureau,
        features=[
            "BUREAU_ACTIVE_COUNT",
            "BUREAU_DAYS_OVERDUE_COUNT",
            "BUREAU_AMOUNT_OVERDUE_COUNT",
        ],
        labels=[
            "Créditos activos",
            "Con días vencidos",
            "Con monto vencido",
        ],
        title="Bureau: comportamiento crediticio por TARGET",
        filename="02_bureau_risk.png",
    )

    plot_feature_comparison(
        analysis=previous,
        features=[
            "PREV_APPROVAL_RATE",
            "PREV_REFUSAL_RATE",
        ],
        labels=[
            "Aprobación",
            "Rechazo",
        ],
        title="Solicitudes anteriores por TARGET",
        filename="03_previous_application_risk.png",
        multiplier=100,
    )

    plot_feature_comparison(
        analysis=pos,
        features=[
            "POS_DPD_RATE",
            "POS_DPD_DEF_RATE",
        ],
        labels=[
            "DPD",
            "DPD DEF",
        ],
        title="POS CASH: morosidad por TARGET",
        filename="04_pos_delinquency.png",
        multiplier=100,
    )

    plot_feature_comparison(
        analysis=credit_card,
        features=[
            "CC_AVG_UTILIZATION",
            "CC_DPD_DEF_RATE",
        ],
        labels=[
            "Utilización",
            "DPD DEF",
        ],
        title="Tarjetas de crédito: comportamiento por TARGET",
        filename="05_credit_card_risk.png",
        multiplier=100,
    )

    plot_feature_comparison(
        analysis=installments,
        features=[
            "INST_LATE_PAYMENT_RATE",
            "INST_UNDERPAYMENT_RATE",
        ],
        labels=[
            "Pago tardío",
            "Pago insuficiente",
        ],
        title="Cuotas: comportamiento de pago por TARGET",
        filename="06_installments_payment_behavior.png",
        multiplier=100,
    )

    plot_feature_comparison(
        analysis=bureau_balance,
        features=[
            "BB_DPD_RATE",
            "BB_SEVERE_DPD_RATE",
        ],
        labels=[
            "DPD",
            "DPD severo",
        ],
        title="Bureau Balance: morosidad por TARGET",
        filename="07_bureau_balance_delinquency.png",
        multiplier=100,
    )

    print("\n" + "=" * 80)
    print("VISUALIZACIONES HISTORICAS GENERADAS CORRECTAMENTE")
    print("=" * 80)


if __name__ == "__main__":
    main()
