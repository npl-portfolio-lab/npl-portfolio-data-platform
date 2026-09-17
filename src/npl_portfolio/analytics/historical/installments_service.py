from pathlib import Path
from typing import Any

import pandas as pd

from npl_portfolio.analytics.historical.duckdb_base_service import (
    DuckDBBaseService,
)
from npl_portfolio.features.installments_features import (
    InstallmentsFeatureBuilder,
)


class InstallmentsService(DuckDBBaseService):
    """
    Servicio especializado para installments_payments.

    La construcción de features históricas se delega a
    InstallmentsFeatureBuilder.

    TARGET se incorpora únicamente después de construir
    las features y se utiliza exclusivamente para análisis EDA.
    """

    FEATURE_COLUMNS = [
        "INST_RECORD_COUNT",
        "INST_CONTRACT_COUNT",
        "INST_AVG_VERSION",
        "INST_MAX_VERSION",
        "INST_AVG_NUMBER",
        "INST_MAX_NUMBER",
        "INST_AVG_INSTALMENT_AMOUNT",
        "INST_MAX_INSTALMENT_AMOUNT",
        "INST_TOTAL_INSTALMENT_AMOUNT",
        "INST_AVG_PAYMENT_AMOUNT",
        "INST_MAX_PAYMENT_AMOUNT",
        "INST_TOTAL_PAYMENT_AMOUNT",
        "INST_AVG_PAYMENT_DELAY",
        "INST_MAX_PAYMENT_DELAY",
        "INST_MIN_PAYMENT_DELAY",
        "INST_LATE_PAYMENT_COUNT",
        "INST_ON_TIME_OR_EARLY_COUNT",
        "INST_MISSING_PAYMENT_DATE_COUNT",
        "INST_AVG_PAYMENT_DIFFERENCE",
        "INST_MAX_PAYMENT_DIFFERENCE",
        "INST_UNDERPAYMENT_COUNT",
        "INST_OVERPAYMENT_COUNT",
        "INST_OLDEST_SCHEDULED_DAY",
        "INST_RECENT_SCHEDULED_DAY",
        "INST_OLDEST_PAYMENT_DAY",
        "INST_RECENT_PAYMENT_DAY",
        "INST_LATE_PAYMENT_RATE",
        "INST_UNDERPAYMENT_RATE",
        "INST_OVERPAYMENT_RATE",
        "INST_PAYMENT_RATIO",
    ]

    def get_installments_target_analysis(
        self,
        application_path: Path,
        target_column: str = "TARGET",
    ) -> dict[str, Any]:
        """
        Construye las features de Installments sin TARGET y
        posteriormente analiza su distribución entre las clases
        de la variable objetivo.
        """
        if not application_path.exists():
            raise FileNotFoundError(f"No existe el archivo: {application_path}")

        feature_builder = InstallmentsFeatureBuilder(
            parquet_path=self.parquet_path,
        )

        features = feature_builder.build()

        application = pd.read_parquet(
            application_path,
            columns=[
                "SK_ID_CURR",
                target_column,
            ],
        )

        dataframe = application.merge(
            features,
            on="SK_ID_CURR",
            how="left",
            validate="one_to_one",
            indicator="_INSTALLMENTS_MERGE",
        )

        dataframe["HAS_INSTALLMENTS_HISTORY"] = (
            dataframe["_INSTALLMENTS_MERGE"] == "both"
        ).astype("int8")

        dataframe = dataframe.drop(
            columns="_INSTALLMENTS_MERGE",
        )

        clients_with_history = int(dataframe["HAS_INSTALLMENTS_HISTORY"].sum())

        clients_without_history = int(len(dataframe) - clients_with_history)

        target_results = []

        target_values = sorted(dataframe[target_column].dropna().unique())

        for target_value in target_values:
            target_data = dataframe.loc[dataframe[target_column] == target_value]

            target_clients = len(target_data)

            target_with_history = int(target_data["HAS_INSTALLMENTS_HISTORY"].sum())

            feature_statistics = []

            for column in self.FEATURE_COLUMNS:
                series = target_data[column].dropna()

                if series.empty:
                    continue

                feature_statistics.append(
                    {
                        "feature": column,
                        "count": int(series.count()),
                        "mean": round(
                            float(series.mean()),
                            4,
                        ),
                        "median": round(
                            float(series.median()),
                            4,
                        ),
                        "p25": round(
                            float(series.quantile(0.25)),
                            4,
                        ),
                        "p75": round(
                            float(series.quantile(0.75)),
                            4,
                        ),
                        "min": round(
                            float(series.min()),
                            4,
                        ),
                        "max": round(
                            float(series.max()),
                            4,
                        ),
                    }
                )

            target_results.append(
                {
                    "target": int(target_value),
                    "clients": int(target_clients),
                    "clients_with_history": (target_with_history),
                    "history_coverage_percentage": round(
                        (
                            target_with_history / target_clients * 100
                            if target_clients
                            else 0.0
                        ),
                        4,
                    ),
                    "features": feature_statistics,
                }
            )

        return {
            "application_clients": int(len(dataframe)),
            "clients_with_history": (clients_with_history),
            "clients_without_history": (clients_without_history),
            "history_coverage_percentage": round(
                (
                    clients_with_history / len(dataframe) * 100
                    if len(dataframe)
                    else 0.0
                ),
                4,
            ),
            "aggregated_features": (self.FEATURE_COLUMNS.copy()),
            "target_statistics": (target_results),
        }
