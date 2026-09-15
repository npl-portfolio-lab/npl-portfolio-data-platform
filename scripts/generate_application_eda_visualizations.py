from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from npl_portfolio.analytics.eda_service import EDAService


PROJECT_ROOT = Path(__file__).resolve().parents[1]

APPLICATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "home_credit"
    / "application_train.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "eda"
    / "application"
)


def save_figure(filename: str) -> None:
    """
    Guarda la figura actual y libera memoria.
    """
    output_path = OUTPUT_DIR / filename

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    print(f"[OK] {output_path}")


def plot_target_distribution(
    service: EDAService,
) -> None:
    """
    Distribución de la variable objetivo de riesgo TARGET.
    """
    result = service.get_target_distribution()

    labels = ["TARGET = 0", "TARGET = 1"]
    values = [
        result["target_0"],
        result["target_1"],
    ]

    percentages = [
        result["target_0_percentage"],
        result["target_1_percentage"],
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(
        labels,
        values,
    )

    ax.set_title("Distribución de TARGET")
    ax.set_ylabel("Número de clientes")

    for bar, percentage in zip(
        bars,
        percentages,
        strict=True,
    ):
        height = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{int(height):,}\n{percentage:.2f}%",
            ha="center",
            va="bottom",
        )

    save_figure("01_target_distribution.png")


def plot_missing_values(
    service: EDAService,
) -> None:
    """
    Top 20 de columnas con mayor porcentaje de valores faltantes.
    """
    result = service.get_missing_values_summary(
        top_n=20,
    )

    data = result["top_missing_columns"]

    dataframe = pd.DataFrame(data)

    dataframe = dataframe.sort_values(
        "null_percentage",
        ascending=True,
    )

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.barh(
        dataframe["column"],
        dataframe["null_percentage"],
    )

    ax.set_title(
        "Top 20 columnas con mayor porcentaje de valores faltantes"
    )
    ax.set_xlabel("Valores faltantes (%)")
    ax.set_ylabel("Variable")

    save_figure("02_missing_values_top20.png")


def plot_numeric_distributions() -> None:
    """
    Distribuciones de variables monetarias seleccionadas.

    Se utiliza el percentil 99 para visualización con el objetivo
    de evitar que valores extremos oculten la forma principal
    de la distribución.

    Los datos originales no son modificados.
    """
    columns = [
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
    ]

    dataframe = pd.read_parquet(
        APPLICATION_PATH,
        columns=columns,
    )

    configurations = [
        (
            "AMT_INCOME_TOTAL",
            "Distribución del ingreso total",
            "03_income_distribution.png",
        ),
        (
            "AMT_CREDIT",
            "Distribución del monto de crédito",
            "04_credit_distribution.png",
        ),
        (
            "AMT_ANNUITY",
            "Distribución de la anualidad",
            "05_annuity_distribution.png",
        ),
    ]

    for column, title, filename in configurations:
        series = dataframe[column].dropna()

        upper_limit = series.quantile(0.99)

        visual_series = series[
            series <= upper_limit
        ]

        fig, ax = plt.subplots(figsize=(9, 5))

        ax.hist(
            visual_series,
            bins=50,
        )

        ax.set_title(
            f"{title}\nVisualización hasta percentil 99"
        )
        ax.set_xlabel(column)
        ax.set_ylabel("Frecuencia")

        save_figure(filename)


def plot_numeric_target_comparisons(
    service: EDAService,
) -> None:
    """
    Compara medianas de variables seleccionadas entre TARGET 0 y 1.
    """
    columns = [
        "DAYS_BIRTH",
        "AMT_CREDIT",
        "AMT_INCOME_TOTAL",
    ]

    result = service.get_numeric_summary_by_target(
        columns=columns,
    )

    configurations = {
        "DAYS_BIRTH": (
            "Edad aproximada por TARGET",
            "06_age_by_target.png",
        ),
        "AMT_CREDIT": (
            "Mediana del crédito por TARGET",
            "07_credit_by_target.png",
        ),
        "AMT_INCOME_TOTAL": (
            "Mediana del ingreso por TARGET",
            "08_income_by_target.png",
        ),
    }

    for statistic in result["statistics"]:
        column = statistic["column"]

        if column not in configurations:
            continue

        title, filename = configurations[column]

        targets = []
        values = []

        for target_statistic in statistic[
            "target_statistics"
        ]:
            target = target_statistic["target"]
            median = target_statistic["median"]

            if column == "DAYS_BIRTH":
                median = abs(median) / 365.25

            targets.append(
                f"TARGET = {target}"
            )
            values.append(median)

        fig, ax = plt.subplots(figsize=(8, 5))

        bars = ax.bar(
            targets,
            values,
        )

        ax.set_title(title)

        if column == "DAYS_BIRTH":
            ax.set_ylabel("Edad aproximada (años)")
        else:
            ax.set_ylabel("Mediana")

        for bar, value in zip(
            bars,
            values,
            strict=True,
        ):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:,.2f}",
                ha="center",
                va="bottom",
            )

        save_figure(filename)


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    service = EDAService(
        parquet_path=APPLICATION_PATH,
    )

    print("=" * 80)
    print("GENERANDO VISUALIZACIONES EDA - APPLICATION_TRAIN")
    print("=" * 80)

    plot_target_distribution(service)
    plot_missing_values(service)
    plot_numeric_distributions()
    plot_numeric_target_comparisons(service)

    print("=" * 80)
    print("VISUALIZACIONES GENERADAS CORRECTAMENTE")
    print("=" * 80)


if __name__ == "__main__":
    main()
