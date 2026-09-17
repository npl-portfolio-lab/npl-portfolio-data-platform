from pathlib import Path
from typing import Any

import pandas as pd

from npl_portfolio.analytics.historical.duckdb_base_service import (
    DuckDBBaseService,
)
from npl_portfolio.features.credit_card_features import (
    CreditCardFeatureBuilder,
)


class CreditCardService(DuckDBBaseService):
    """
    Servicio especializado para credit_card_balance.

    La construcción de features históricas se delega a
    CreditCardFeatureBuilder.

    Este servicio incorpora TARGET únicamente después de
    construir las features, con fines de análisis EDA.
    """

    FEATURE_COLUMNS = [
        "CC_RECORD_COUNT",
        "CC_CONTRACT_COUNT",
        "CC_ACTIVE_RECORD_COUNT",
        "CC_COMPLETED_RECORD_COUNT",
        "CC_AVG_BALANCE",
        "CC_MAX_BALANCE",
        "CC_AVG_CREDIT_LIMIT",
        "CC_MAX_CREDIT_LIMIT",
        "CC_AVG_DRAWINGS",
        "CC_TOTAL_DRAWINGS",
        "CC_AVG_PAYMENT_CURRENT",
        "CC_AVG_PAYMENT_TOTAL",
        "CC_TOTAL_PAYMENT",
        "CC_AVG_RECEIVABLE",
        "CC_AVG_DRAWING_COUNT",
        "CC_AVG_MATURE_INSTALMENTS",
        "CC_DPD_RECORD_COUNT",
        "CC_DPD_DEF_RECORD_COUNT",
        "CC_MAX_DPD",
        "CC_AVG_DPD",
        "CC_MAX_DPD_DEF",
        "CC_AVG_DPD_DEF",
        "CC_OLDEST_MONTH",
        "CC_RECENT_MONTH",
        "CC_AVG_UTILIZATION",
        "CC_MAX_UTILIZATION",
        "CC_DPD_RATE",
        "CC_DPD_DEF_RATE",
    ]

    def get_credit_card_target_analysis(
        self,
        application_path: Path,
        target_column: str = "TARGET",
    ) -> dict[str, Any]:
        """
        Construye las features de Credit Card sin TARGET y
        posteriormente analiza su distribución entre las clases
        de la variable objetivo.
        """
        if not application_path.exists():
            raise FileNotFoundError(f"No existe el archivo: {application_path}")

        feature_builder = CreditCardFeatureBuilder(
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
            indicator="_CC_MERGE",
        )

        dataframe["HAS_CREDIT_CARD_HISTORY"] = (
            dataframe["_CC_MERGE"] == "both"
        ).astype("int8")

        dataframe = dataframe.drop(
            columns="_CC_MERGE",
        )

        clients_with_history = int(dataframe["HAS_CREDIT_CARD_HISTORY"].sum())

        clients_without_history = int(len(dataframe) - clients_with_history)

        target_results = []

        target_values = sorted(dataframe[target_column].dropna().unique())

        for target_value in target_values:
            target_data = dataframe.loc[dataframe[target_column] == target_value]

            target_clients = len(target_data)

            target_with_history = int(target_data["HAS_CREDIT_CARD_HISTORY"].sum())

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
