from pathlib import Path
from typing import Any

import pandas as pd

from npl_portfolio.analytics.historical.duckdb_base_service import (
    DuckDBBaseService,
)
from npl_portfolio.features.bureau_balance_features import (
    BureauBalanceFeatureBuilder,
)


class BureauBalanceService(DuckDBBaseService):
    """
    Servicio especializado para bureau_balance.

    bureau_balance no contiene SK_ID_CURR.

    La relación necesaria es:

        bureau_balance.SK_ID_BUREAU
                    |
                    v
        bureau.SK_ID_BUREAU
                    |
                    v
        bureau.SK_ID_CURR
                    |
                    v
        application_train.SK_ID_CURR

    La construcción de features se delega a
    BureauBalanceFeatureBuilder.

    TARGET se incorpora únicamente después de construir
    las features y se utiliza exclusivamente para análisis EDA.
    """

    FEATURE_COLUMNS = [
        "BB_RECORD_COUNT",
        "BB_BUREAU_CREDIT_COUNT",
        "BB_OLDEST_MONTH",
        "BB_RECENT_MONTH",
        "BB_STATUS_0_COUNT",
        "BB_STATUS_1_COUNT",
        "BB_STATUS_2_COUNT",
        "BB_STATUS_3_COUNT",
        "BB_STATUS_4_COUNT",
        "BB_STATUS_5_COUNT",
        "BB_STATUS_C_COUNT",
        "BB_STATUS_X_COUNT",
        "BB_DPD_RECORD_COUNT",
        "BB_SEVERE_DPD_RECORD_COUNT",
        "BB_MAX_STATUS_LEVEL",
        "BB_DPD_RATE",
        "BB_SEVERE_DPD_RATE",
        "BB_CLOSED_STATUS_RATE",
        "BB_UNKNOWN_STATUS_RATE",
    ]

    def get_mapping_diagnostics(
        self,
        bureau_path: Path,
    ) -> dict[str, Any]:
        """
        Cuantifica la cobertura de SK_ID_BUREAU entre
        bureau_balance y bureau.
        """
        if not bureau_path.exists():
            raise FileNotFoundError(f"No existe el archivo: {bureau_path}")

        connection = self._get_connection()

        try:
            query = """
                SELECT
                    COUNT(*) AS total_rows,

                    SUM(
                        CASE
                            WHEN bureau.SK_ID_BUREAU IS NOT NULL
                            THEN 1
                            ELSE 0
                        END
                    ) AS mapped_rows,

                    SUM(
                        CASE
                            WHEN bureau.SK_ID_BUREAU IS NULL
                            THEN 1
                            ELSE 0
                        END
                    ) AS unmapped_rows,

                    COUNT(
                        DISTINCT CASE
                            WHEN bureau.SK_ID_BUREAU IS NULL
                            THEN bureau_balance.SK_ID_BUREAU
                        END
                    ) AS unmapped_bureau_ids

                FROM read_parquet(?) AS bureau_balance

                LEFT JOIN read_parquet(?) AS bureau
                    ON bureau_balance.SK_ID_BUREAU
                    = bureau.SK_ID_BUREAU
            """

            result = connection.execute(
                query,
                [
                    str(self.parquet_path),
                    str(bureau_path),
                ],
            ).fetchone()

        finally:
            connection.close()

        total_rows = int(result[0])
        mapped_rows = int(result[1] or 0)
        unmapped_rows = int(result[2] or 0)
        unmapped_bureau_ids = int(result[3] or 0)

        return {
            "total_rows": total_rows,
            "mapped_rows": mapped_rows,
            "unmapped_rows": unmapped_rows,
            "unmapped_bureau_ids": unmapped_bureau_ids,
            "mapped_percentage": round(
                (mapped_rows / total_rows * 100 if total_rows else 0.0),
                4,
            ),
            "unmapped_percentage": round(
                (unmapped_rows / total_rows * 100 if total_rows else 0.0),
                4,
            ),
        }

    def get_bureau_balance_target_analysis(
        self,
        bureau_path: Path,
        application_path: Path,
        target_column: str = "TARGET",
    ) -> dict[str, Any]:
        """
        Construye las features de bureau_balance sin TARGET
        y posteriormente analiza su distribución entre las
        clases de la variable objetivo.
        """
        if not bureau_path.exists():
            raise FileNotFoundError(f"No existe el archivo: {bureau_path}")

        if not application_path.exists():
            raise FileNotFoundError(f"No existe el archivo: {application_path}")

        mapping_diagnostics = self.get_mapping_diagnostics(
            bureau_path=bureau_path,
        )

        feature_builder = BureauBalanceFeatureBuilder(
            parquet_path=self.parquet_path,
            bureau_path=bureau_path,
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
            indicator="_BUREAU_BALANCE_MERGE",
        )

        dataframe["HAS_BUREAU_BALANCE_HISTORY"] = (
            dataframe["_BUREAU_BALANCE_MERGE"] == "both"
        ).astype("int8")

        dataframe = dataframe.drop(
            columns="_BUREAU_BALANCE_MERGE",
        )

        clients_with_history = int(dataframe["HAS_BUREAU_BALANCE_HISTORY"].sum())

        clients_without_history = int(len(dataframe) - clients_with_history)

        target_results = []

        target_values = sorted(dataframe[target_column].dropna().unique())

        for target_value in target_values:
            target_data = dataframe.loc[dataframe[target_column] == target_value]

            target_clients = len(target_data)

            target_with_history = int(target_data["HAS_BUREAU_BALANCE_HISTORY"].sum())

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
            "mapping_diagnostics": (mapping_diagnostics),
            "aggregated_features": (self.FEATURE_COLUMNS.copy()),
            "target_statistics": (target_results),
        }
