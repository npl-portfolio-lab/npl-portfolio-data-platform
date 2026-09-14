from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


class HistoricalEDAService:
    """
    Servicio reutilizable para análisis exploratorio
    de tablas históricas de Home Credit.
    """

    def __init__(
        self,
        parquet_path: Path,
    ) -> None:
        self.parquet_path = parquet_path

        if not self.parquet_path.exists():
            raise FileNotFoundError(f"No existe el archivo: " f"{self.parquet_path}")

    def get_dataset_overview(
        self,
    ) -> dict[str, Any]:
        """
        Obtiene información estructural del Parquet
        usando metadata sin cargar todo el dataset.
        """

        parquet_file = pq.ParquetFile(self.parquet_path)

        metadata = parquet_file.metadata

        return {
            "file": self.parquet_path.name,
            "rows": metadata.num_rows,
            "columns": metadata.num_columns,
            "row_groups": metadata.num_row_groups,
        }

    def get_unique_clients(
        self,
        client_column: str = "SK_ID_CURR",
    ) -> dict[str, Any]:
        """
        Calcula la cantidad de clientes únicos
        presentes en la tabla histórica.
        """

        dataframe = pd.read_parquet(
            self.parquet_path,
            columns=[client_column],
        )

        total_rows = len(dataframe)

        unique_clients = int(dataframe[client_column].nunique())

        average_records = total_rows / unique_clients if unique_clients else 0.0

        return {
            "client_column": client_column,
            "total_rows": total_rows,
            "unique_clients": unique_clients,
            "average_records_per_client": round(
                average_records,
                4,
            ),
        }

    def get_categorical_summary(
        self,
        columns: list[str],
        top_n: int = 10,
    ) -> dict[str, Any]:
        """
        Obtiene distribución de variables categóricas.
        """

        if not columns:
            raise ValueError("Debe proporcionar al menos una columna.")

        dataframe = pd.read_parquet(
            self.parquet_path,
            columns=columns,
        )

        results = []

        for column in columns:
            series = dataframe[column]

            total_rows = len(series)

            null_count = int(series.isna().sum())

            counts = series.value_counts(dropna=True).head(top_n)

            categories = []

            for value, count in counts.items():
                percentage = count / total_rows * 100 if total_rows else 0.0

                categories.append(
                    {
                        "value": str(value),
                        "count": int(count),
                        "percentage": round(
                            percentage,
                            4,
                        ),
                    }
                )

            results.append(
                {
                    "column": column,
                    "unique_values": int(series.nunique(dropna=True)),
                    "null_count": null_count,
                    "null_percentage": round(
                        (null_count / total_rows * 100) if total_rows else 0.0,
                        4,
                    ),
                    "top_categories": categories,
                }
            )

        return {
            "columns_analyzed": len(results),
            "statistics": results,
        }

    def get_numeric_summary(
        self,
        columns: list[str],
    ) -> dict[str, Any]:
        """
        Obtiene estadísticas descriptivas
        de variables numéricas.
        """

        if not columns:
            raise ValueError("Debe proporcionar al menos una columna.")

        dataframe = pd.read_parquet(
            self.parquet_path,
            columns=columns,
        )

        results = []

        for column in columns:
            series = dataframe[column]

            valid = series.dropna()

            null_count = int(series.isna().sum())

            results.append(
                {
                    "column": column,
                    "count": int(valid.count()),
                    "null_count": null_count,
                    "null_percentage": round(
                        (null_count / len(series) * 100) if len(series) else 0.0,
                        4,
                    ),
                    "mean": round(
                        float(valid.mean()),
                        4,
                    ),
                    "median": round(
                        float(valid.median()),
                        4,
                    ),
                    "std": round(
                        float(valid.std()),
                        4,
                    ),
                    "min": round(
                        float(valid.min()),
                        4,
                    ),
                    "p25": round(
                        float(valid.quantile(0.25)),
                        4,
                    ),
                    "p75": round(
                        float(valid.quantile(0.75)),
                        4,
                    ),
                    "max": round(
                        float(valid.max()),
                        4,
                    ),
                }
            )

        return {
            "columns_analyzed": len(results),
            "statistics": results,
        }

    def get_records_per_client_summary(
        self,
        client_column: str = "SK_ID_CURR",
    ) -> dict[str, Any]:
        """
        Analiza cuántos registros históricos
        tiene cada cliente.
        """

        dataframe = pd.read_parquet(
            self.parquet_path,
            columns=[client_column],
        )

        counts = dataframe[client_column].value_counts()

        return {
            "clients": int(counts.count()),
            "mean_records": round(
                float(counts.mean()),
                4,
            ),
            "median_records": round(
                float(counts.median()),
                4,
            ),
            "min_records": int(counts.min()),
            "p25_records": round(
                float(counts.quantile(0.25)),
                4,
            ),
            "p75_records": round(
                float(counts.quantile(0.75)),
                4,
            ),
            "p90_records": round(
                float(counts.quantile(0.90)),
                4,
            ),
            "p95_records": round(
                float(counts.quantile(0.95)),
                4,
            ),
            "p99_records": round(
                float(counts.quantile(0.99)),
                4,
            ),
            "max_records": int(counts.max()),
        }

    def get_bureau_target_analysis(
        self,
        application_path: Path,
        target_column: str = "TARGET",
    ) -> dict[str, Any]:
        """
        Agrega el historial de bureau por cliente y compara
        las métricas resultantes entre las clases de TARGET.

        La unidad de análisis final es SK_ID_CURR,
        no cada crédito individual.
        """

        if not application_path.exists():
            raise FileNotFoundError(f"No existe el archivo: " f"{application_path}")

        bureau_columns = [
            "SK_ID_CURR",
            "CREDIT_ACTIVE",
            "CREDIT_DAY_OVERDUE",
            "DAYS_CREDIT",
            "AMT_CREDIT_SUM",
            "AMT_CREDIT_SUM_DEBT",
            "AMT_CREDIT_SUM_OVERDUE",
        ]

        bureau = pd.read_parquet(
            self.parquet_path,
            columns=bureau_columns,
        )

        application = pd.read_parquet(
            application_path,
            columns=[
                "SK_ID_CURR",
                target_column,
            ],
        )

        bureau["IS_ACTIVE"] = bureau["CREDIT_ACTIVE"].eq("Active").astype("int8")

        bureau["IS_CLOSED"] = bureau["CREDIT_ACTIVE"].eq("Closed").astype("int8")

        bureau["HAS_DAYS_OVERDUE"] = bureau["CREDIT_DAY_OVERDUE"].gt(0).astype("int8")

        bureau["HAS_AMOUNT_OVERDUE"] = (
            bureau["AMT_CREDIT_SUM_OVERDUE"].gt(0).astype("int8")
        )

        bureau_aggregated = bureau.groupby(
            "SK_ID_CURR",
            as_index=False,
        ).agg(
            BUREAU_CREDIT_COUNT=(
                "SK_ID_CURR",
                "size",
            ),
            BUREAU_ACTIVE_COUNT=(
                "IS_ACTIVE",
                "sum",
            ),
            BUREAU_CLOSED_COUNT=(
                "IS_CLOSED",
                "sum",
            ),
            BUREAU_DAYS_OVERDUE_COUNT=(
                "HAS_DAYS_OVERDUE",
                "sum",
            ),
            BUREAU_AMOUNT_OVERDUE_COUNT=(
                "HAS_AMOUNT_OVERDUE",
                "sum",
            ),
            BUREAU_AVG_DAYS_CREDIT=(
                "DAYS_CREDIT",
                "mean",
            ),
            BUREAU_MIN_DAYS_CREDIT=(
                "DAYS_CREDIT",
                "min",
            ),
            BUREAU_TOTAL_CREDIT=(
                "AMT_CREDIT_SUM",
                "sum",
            ),
            BUREAU_TOTAL_DEBT=(
                "AMT_CREDIT_SUM_DEBT",
                "sum",
            ),
            BUREAU_TOTAL_OVERDUE=(
                "AMT_CREDIT_SUM_OVERDUE",
                "sum",
            ),
            BUREAU_MAX_OVERDUE_DAYS=(
                "CREDIT_DAY_OVERDUE",
                "max",
            ),
        )

        merged = application.merge(
            bureau_aggregated,
            on="SK_ID_CURR",
            how="left",
            validate="one_to_one",
        )

        feature_columns = [
            column for column in bureau_aggregated.columns if column != "SK_ID_CURR"
        ]

        clients_with_bureau = int(merged["BUREAU_CREDIT_COUNT"].notna().sum())

        clients_without_bureau = int(merged["BUREAU_CREDIT_COUNT"].isna().sum())

        target_results = []

        for target_value in sorted(merged[target_column].dropna().unique()):
            target_data = merged.loc[merged[target_column] == target_value]

            target_clients = len(target_data)

            target_with_bureau = int(target_data["BUREAU_CREDIT_COUNT"].notna().sum())

            feature_statistics = []

            for column in feature_columns:
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
                    "clients_with_bureau": (target_with_bureau),
                    "bureau_coverage_percentage": round(
                        (
                            (target_with_bureau / target_clients * 100)
                            if target_clients
                            else 0.0
                        ),
                        4,
                    ),
                    "features": (feature_statistics),
                }
            )

        return {
            "application_clients": int(len(application)),
            "clients_with_bureau": (clients_with_bureau),
            "clients_without_bureau": (clients_without_bureau),
            "bureau_coverage_percentage": round(
                (
                    (clients_with_bureau / len(application) * 100)
                    if len(application)
                    else 0.0
                ),
                4,
            ),
            "aggregated_features": (feature_columns),
            "target_statistics": (target_results),
        }

    def get_previous_application_target_analysis(
        self,
        application_path: Path,
        target_column: str = "TARGET",
    ) -> dict[str, Any]:
        """
        Agrega previous_application por cliente y compara
        las características históricas entre TARGET=0 y TARGET=1.

        La unidad final de análisis es SK_ID_CURR.
        """

        if not application_path.exists():
            raise FileNotFoundError(f"No existe el archivo: {application_path}")

        previous_columns = [
            "SK_ID_CURR",
            "NAME_CONTRACT_STATUS",
            "AMT_APPLICATION",
            "AMT_CREDIT",
            "AMT_ANNUITY",
            "CNT_PAYMENT",
            "DAYS_DECISION",
        ]

        previous = pd.read_parquet(
            self.parquet_path,
            columns=previous_columns,
        )

        application = pd.read_parquet(
            application_path,
            columns=[
                "SK_ID_CURR",
                target_column,
            ],
        )

        # -------------------------------------------------
        # Indicadores de estado
        # -------------------------------------------------

        previous["IS_APPROVED"] = (
            previous["NAME_CONTRACT_STATUS"].eq("Approved").astype("int8")
        )

        previous["IS_REFUSED"] = (
            previous["NAME_CONTRACT_STATUS"].eq("Refused").astype("int8")
        )

        previous["IS_CANCELED"] = (
            previous["NAME_CONTRACT_STATUS"].eq("Canceled").astype("int8")
        )

        previous["IS_UNUSED"] = (
            previous["NAME_CONTRACT_STATUS"].eq("Unused offer").astype("int8")
        )

        # Diferencia entre lo solicitado y lo concedido.
        previous["CREDIT_DIFFERENCE"] = (
            previous["AMT_APPLICATION"] - previous["AMT_CREDIT"]
        )

        # -------------------------------------------------
        # Agregación por cliente
        # -------------------------------------------------

        previous_aggregated = previous.groupby(
            "SK_ID_CURR",
            as_index=False,
        ).agg(
            PREV_APPLICATION_COUNT=(
                "SK_ID_CURR",
                "size",
            ),
            PREV_APPROVED_COUNT=(
                "IS_APPROVED",
                "sum",
            ),
            PREV_REFUSED_COUNT=(
                "IS_REFUSED",
                "sum",
            ),
            PREV_CANCELED_COUNT=(
                "IS_CANCELED",
                "sum",
            ),
            PREV_UNUSED_COUNT=(
                "IS_UNUSED",
                "sum",
            ),
            PREV_AVG_APPLICATION_AMOUNT=(
                "AMT_APPLICATION",
                "mean",
            ),
            PREV_AVG_CREDIT_AMOUNT=(
                "AMT_CREDIT",
                "mean",
            ),
            PREV_TOTAL_APPLICATION_AMOUNT=(
                "AMT_APPLICATION",
                "sum",
            ),
            PREV_TOTAL_CREDIT_AMOUNT=(
                "AMT_CREDIT",
                "sum",
            ),
            PREV_AVG_ANNUITY=(
                "AMT_ANNUITY",
                "mean",
            ),
            PREV_AVG_PAYMENT_COUNT=(
                "CNT_PAYMENT",
                "mean",
            ),
            PREV_AVG_CREDIT_DIFFERENCE=(
                "CREDIT_DIFFERENCE",
                "mean",
            ),
            PREV_AVG_DAYS_DECISION=(
                "DAYS_DECISION",
                "mean",
            ),
            PREV_LAST_DECISION_DAYS=(
                "DAYS_DECISION",
                "max",
            ),
        )

        # -------------------------------------------------
        # Tasas
        # -------------------------------------------------

        previous_aggregated["PREV_APPROVAL_RATE"] = (
            previous_aggregated["PREV_APPROVED_COUNT"]
            / previous_aggregated["PREV_APPLICATION_COUNT"]
        )

        previous_aggregated["PREV_REFUSAL_RATE"] = (
            previous_aggregated["PREV_REFUSED_COUNT"]
            / previous_aggregated["PREV_APPLICATION_COUNT"]
        )

        previous_aggregated["PREV_CANCELLATION_RATE"] = (
            previous_aggregated["PREV_CANCELED_COUNT"]
            / previous_aggregated["PREV_APPLICATION_COUNT"]
        )

        # -------------------------------------------------
        # Join con application_train
        # -------------------------------------------------

        merged = application.merge(
            previous_aggregated,
            on="SK_ID_CURR",
            how="left",
            validate="one_to_one",
        )

        feature_columns = [
            column for column in previous_aggregated.columns if column != "SK_ID_CURR"
        ]

        clients_with_history = int(merged["PREV_APPLICATION_COUNT"].notna().sum())

        clients_without_history = int(merged["PREV_APPLICATION_COUNT"].isna().sum())

        target_results = []

        for target_value in sorted(merged[target_column].dropna().unique()):
            target_data = merged.loc[merged[target_column] == target_value]

            target_clients = len(target_data)

            target_with_history = int(
                target_data["PREV_APPLICATION_COUNT"].notna().sum()
            )

            feature_statistics = []

            for column in feature_columns:
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
                            (target_with_history / target_clients * 100)
                            if target_clients
                            else 0.0
                        ),
                        4,
                    ),
                    "features": (feature_statistics),
                }
            )

        return {
            "application_clients": int(len(application)),
            "clients_with_history": (clients_with_history),
            "clients_without_history": (clients_without_history),
            "history_coverage_percentage": round(
                (
                    (clients_with_history / len(application) * 100)
                    if len(application)
                    else 0.0
                ),
                4,
            ),
            "aggregated_features": (feature_columns),
            "target_statistics": (target_results),
        }
