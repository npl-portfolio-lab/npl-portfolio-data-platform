from pathlib import Path
from typing import Any

import pandas as pd

from npl_portfolio.analytics.historical.duckdb_base_service import (
    DuckDBBaseService,
)
from npl_portfolio.features.pos_cash_features import (
    POSCashFeatureBuilder,
)


class POSCashService(DuckDBBaseService):
    """
    Servicio especializado para el análisis de POS_CASH_balance.

    Hereda las operaciones descriptivas generales de
    DuckDBBaseService.

    La construcción de features históricas se delega a
    POSCashFeatureBuilder. Este servicio únicamente incorpora
    la variable objetivo para realizar análisis EDA.
    """

    FEATURE_COLUMNS = [
        "POS_RECORD_COUNT",
        "POS_CONTRACT_COUNT",
        "POS_ACTIVE_RECORD_COUNT",
        "POS_COMPLETED_RECORD_COUNT",
        "POS_DPD_RECORD_COUNT",
        "POS_DPD_DEF_RECORD_COUNT",
        "POS_MAX_DPD",
        "POS_AVG_DPD",
        "POS_MAX_DPD_DEF",
        "POS_AVG_DPD_DEF",
        "POS_AVG_INSTALMENT",
        "POS_AVG_INSTALMENT_FUTURE",
        "POS_OLDEST_MONTH",
        "POS_RECENT_MONTH",
        "POS_DPD_RATE",
        "POS_DPD_DEF_RATE",
    ]

    def get_pos_cash_target_analysis(
        self,
        application_path: Path,
        target_column: str = "TARGET",
    ) -> dict[str, Any]:
        """
        Construye las features POS independientemente de TARGET
        y posteriormente analiza su distribución entre las clases
        de la variable objetivo.
        """
        if not application_path.exists():
            raise FileNotFoundError(f"No existe el archivo: {application_path}")

        feature_builder = POSCashFeatureBuilder(
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
            indicator="_POS_MERGE",
        )

        dataframe["HAS_POS_HISTORY"] = (dataframe["_POS_MERGE"] == "both").astype(
            "int8"
        )

        dataframe = dataframe.drop(
            columns="_POS_MERGE",
        )

        clients_with_history = int(dataframe["HAS_POS_HISTORY"].sum())

        clients_without_history = len(dataframe) - clients_with_history

        target_results = []

        target_values = sorted(dataframe[target_column].dropna().unique())

        for target_value in target_values:
            target_data = dataframe.loc[dataframe[target_column] == target_value]

            target_clients = len(target_data)

            target_with_history = int(target_data["HAS_POS_HISTORY"].sum())

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
            "clients_without_history": int(clients_without_history),
            "history_coverage_percentage": round(
                (
                    clients_with_history / len(dataframe) * 100
                    if len(dataframe)
                    else 0.0
                ),
                4,
            ),
            "aggregated_features": (self.FEATURE_COLUMNS.copy()),
            "target_statistics": target_results,
        }
