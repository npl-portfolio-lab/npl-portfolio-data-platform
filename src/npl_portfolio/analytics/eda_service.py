from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


class EDAService:
    """
    Servicio encargado de obtener métricas exploratorias
    sobre los datasets procesados de Home Credit.
    """

    def __init__(
        self,
        parquet_path: Path,
    ) -> None:
        if not parquet_path.exists():
            raise FileNotFoundError(f"No existe el archivo Parquet: {parquet_path}")

        self.parquet_path = parquet_path

    def get_dataset_overview(
        self,
    ) -> dict[str, Any]:
        """
        Obtiene información general del dataset utilizando
        metadata de Parquet sin cargar todo el archivo.
        """

        parquet_file = pq.ParquetFile(self.parquet_path)

        metadata = parquet_file.metadata

        return {
            "file": self.parquet_path.name,
            "rows": metadata.num_rows,
            "columns": metadata.num_columns,
            "row_groups": metadata.num_row_groups,
        }

    def get_target_distribution(
        self,
    ) -> dict[str, Any]:
        """
        Calcula la distribución de TARGET.
        """

        dataframe = pd.read_parquet(
            self.parquet_path,
            columns=["TARGET"],
        )

        counts = dataframe["TARGET"].value_counts(dropna=False).sort_index()

        total = len(dataframe)

        target_0 = int(counts.get(0, 0))

        target_1 = int(counts.get(1, 0))

        target_0_percentage = (target_0 / total) * 100 if total else 0.0

        target_1_percentage = (target_1 / total) * 100 if total else 0.0

        return {
            "total_clients": total,
            "target_0": target_0,
            "target_1": target_1,
            "target_0_percentage": round(
                target_0_percentage,
                4,
            ),
            "target_1_percentage": round(
                target_1_percentage,
                4,
            ),
        }

    def get_column_types(
        self,
    ) -> dict[str, Any]:
        """
        Obtiene los tipos de datos almacenados
        en el archivo Parquet.
        """

        parquet_file = pq.ParquetFile(self.parquet_path)

        schema = parquet_file.schema_arrow

        columns = []

        for field in schema:
            columns.append(
                {
                    "column": field.name,
                    "dtype": str(field.type),
                }
            )

        return {
            "total_columns": len(columns),
            "columns": columns,
        }

    def get_missing_values_summary(
        self,
        top_n: int = 20,
    ) -> dict[str, Any]:
        """
        Analiza los valores faltantes utilizando
        batches de Parquet.
        """

        if top_n <= 0:
            raise ValueError("top_n debe ser mayor que cero.")

        parquet_file = pq.ParquetFile(self.parquet_path)

        total_rows = parquet_file.metadata.num_rows

        column_names = parquet_file.schema_arrow.names

        null_counts = {column: 0 for column in column_names}

        for batch in parquet_file.iter_batches(batch_size=100_000):
            for index, column in enumerate(batch.schema.names):
                null_counts[column] += batch.column(index).null_count

        column_results = []

        for column in column_names:
            null_count = null_counts[column]

            null_percentage = (null_count / total_rows) * 100 if total_rows else 0.0

            column_results.append(
                {
                    "column": column,
                    "null_count": null_count,
                    "null_percentage": round(
                        null_percentage,
                        4,
                    ),
                }
            )

        column_results.sort(
            key=lambda item: item["null_percentage"],
            reverse=True,
        )

        columns_with_missing = [
            item for item in column_results if item["null_count"] > 0
        ]

        over_30_percent = sum(item["null_percentage"] >= 30 for item in column_results)

        over_50_percent = sum(item["null_percentage"] >= 50 for item in column_results)

        return {
            "total_rows": total_rows,
            "total_columns": len(column_names),
            "columns_with_missing": len(columns_with_missing),
            "columns_without_missing": (len(column_names) - len(columns_with_missing)),
            "columns_over_30_percent": (over_30_percent),
            "columns_over_50_percent": (over_50_percent),
            "top_missing_columns": (columns_with_missing[:top_n]),
        }

    def get_numeric_summary(
        self,
        columns: list[str],
    ) -> dict[str, Any]:
        """
        Calcula estadísticas descriptivas para un conjunto
        seleccionado de variables numéricas.

        Solo carga las columnas solicitadas.
        """

        if not columns:
            raise ValueError("Debe proporcionar al menos una columna.")

        parquet_file = pq.ParquetFile(self.parquet_path)

        available_columns = set(parquet_file.schema_arrow.names)

        missing_columns = [
            column for column in columns if column not in available_columns
        ]

        if missing_columns:
            raise ValueError(
                "Las siguientes columnas no existen: " + ", ".join(missing_columns)
            )

        dataframe = pd.read_parquet(
            self.parquet_path,
            columns=columns,
        )

        results = []

        for column in columns:
            series = dataframe[column]

            if not pd.api.types.is_numeric_dtype(series):
                continue

            valid_values = series.dropna()

            total_rows = len(series)
            valid_count = len(valid_values)
            null_count = int(series.isna().sum())

            null_percentage = (null_count / total_rows) * 100 if total_rows else 0.0

            if valid_count == 0:
                results.append(
                    {
                        "column": column,
                        "count": 0,
                        "null_count": null_count,
                        "null_percentage": round(
                            null_percentage,
                            4,
                        ),
                        "mean": None,
                        "median": None,
                        "std": None,
                        "min": None,
                        "p25": None,
                        "p75": None,
                        "max": None,
                    }
                )
                continue

            results.append(
                {
                    "column": column,
                    "count": valid_count,
                    "null_count": null_count,
                    "null_percentage": round(
                        null_percentage,
                        4,
                    ),
                    "mean": round(
                        float(valid_values.mean()),
                        4,
                    ),
                    "median": round(
                        float(valid_values.median()),
                        4,
                    ),
                    "std": round(
                        float(valid_values.std()),
                        4,
                    ),
                    "min": round(
                        float(valid_values.min()),
                        4,
                    ),
                    "p25": round(
                        float(valid_values.quantile(0.25)),
                        4,
                    ),
                    "p75": round(
                        float(valid_values.quantile(0.75)),
                        4,
                    ),
                    "max": round(
                        float(valid_values.max()),
                        4,
                    ),
                }
            )

        return {
            "columns_analyzed": len(results),
            "statistics": results,
        }

    def get_value_frequency(
        self,
        column: str,
        value: int | float,
    ) -> dict[str, Any]:
        """
        Calcula cuántas veces aparece un valor específico
        dentro de una columna.
        """

        dataframe = pd.read_parquet(
            self.parquet_path,
            columns=[column],
        )

        total_rows = len(dataframe)

        occurrences = int((dataframe[column] == value).sum())

        percentage = (occurrences / total_rows) * 100 if total_rows else 0.0

        return {
            "column": column,
            "value": value,
            "occurrences": occurrences,
            "percentage": round(
                percentage,
                4,
            ),
        }

    def get_percentiles(
        self,
        column: str,
        percentiles: list[float],
    ) -> dict[str, Any]:
        """
        Calcula percentiles específicos para
        una variable numérica.
        """

        dataframe = pd.read_parquet(
            self.parquet_path,
            columns=[column],
        )

        series = dataframe[column].dropna()

        results = {}

        for percentile in percentiles:
            if percentile < 0 or percentile > 1:
                raise ValueError("Los percentiles deben estar " "entre 0 y 1.")

            value = series.quantile(percentile)

            results[f"p{percentile * 100:g}"] = round(
                float(value),
                4,
            )

        return {
            "column": column,
            "percentiles": results,
        }

    def get_categorical_summary(
        self,
        columns: list[str],
        top_n: int = 10,
    ) -> dict[str, Any]:
        """
        Analiza la distribución de variables categóricas.

        Retorna las categorías más frecuentes,
        cantidades, porcentajes y valores faltantes.
        """

        if not columns:
            raise ValueError("Debe proporcionar al menos una columna.")

        if top_n <= 0:
            raise ValueError("top_n debe ser mayor que cero.")

        parquet_file = pq.ParquetFile(self.parquet_path)

        available_columns = set(parquet_file.schema_arrow.names)

        missing_columns = [
            column for column in columns if column not in available_columns
        ]

        if missing_columns:
            raise ValueError(
                "Las siguientes columnas no existen: " + ", ".join(missing_columns)
            )

        dataframe = pd.read_parquet(
            self.parquet_path,
            columns=columns,
        )

        results = []

        for column in columns:
            series = dataframe[column]

            total_rows = len(series)

            null_count = int(series.isna().sum())

            null_percentage = (null_count / total_rows) * 100 if total_rows else 0.0

            counts = series.value_counts(dropna=True).head(top_n)

            categories = []

            for value, count in counts.items():
                percentage = (count / total_rows) * 100 if total_rows else 0.0

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
                        null_percentage,
                        4,
                    ),
                    "top_categories": categories,
                }
            )

        return {
            "columns_analyzed": len(results),
            "statistics": results,
        }

    def get_target_rate_by_category(
        self,
        columns: list[str],
        target_column: str = "TARGET",
        top_n: int = 20,
    ) -> dict[str, Any]:
        """
        Calcula la tasa de TARGET = 1 para cada categoría.

        Permite identificar categorías con diferentes
        niveles observados de riesgo en el dataset.
        """

        if not columns:
            raise ValueError("Debe proporcionar al menos una columna.")

        if top_n <= 0:
            raise ValueError("top_n debe ser mayor que cero.")

        required_columns = list(dict.fromkeys(columns + [target_column]))

        dataframe = pd.read_parquet(
            self.parquet_path,
            columns=required_columns,
        )

        results = []

        for column in columns:
            working = dataframe[[column, target_column]].copy()

            working[column] = working[column].astype("string").fillna("<MISSING>")

            grouped = (
                working.groupby(
                    column,
                    dropna=False,
                )[target_column]
                .agg(
                    clients="count",
                    target_1="sum",
                    target_rate="mean",
                )
                .reset_index()
            )

            grouped["target_0"] = grouped["clients"] - grouped["target_1"]

            grouped = grouped.sort_values(
                by="clients",
                ascending=False,
            ).head(top_n)

            categories = []

            for _, row in grouped.iterrows():
                categories.append(
                    {
                        "value": str(row[column]),
                        "clients": int(row["clients"]),
                        "target_0": int(row["target_0"]),
                        "target_1": int(row["target_1"]),
                        "target_rate": round(
                            float(row["target_rate"]) * 100,
                            4,
                        ),
                    }
                )

            results.append(
                {
                    "column": column,
                    "categories": categories,
                }
            )

        return {
            "target_column": target_column,
            "columns_analyzed": len(results),
            "statistics": results,
        }

    def get_numeric_summary_by_target(
        self,
        columns: list[str],
        target_column: str = "TARGET",
    ) -> dict[str, Any]:
        """
        Compara variables numéricas entre las clases
        de la variable objetivo.

        Para cada variable y cada valor de TARGET calcula:
        cantidad, media, mediana, percentiles 25 y 75,
        mínimo y máximo.
        """

        if not columns:
            raise ValueError("Debe proporcionar al menos una columna.")

        parquet_file = pq.ParquetFile(self.parquet_path)

        available_columns = set(parquet_file.schema_arrow.names)

        required_columns = columns + [target_column]

        missing_columns = [
            column for column in required_columns if column not in available_columns
        ]

        if missing_columns:
            raise ValueError(
                "Las siguientes columnas no existen: " + ", ".join(missing_columns)
            )

        dataframe = pd.read_parquet(
            self.parquet_path,
            columns=required_columns,
        )

        results = []

        for column in columns:
            target_statistics = []

            for target_value in sorted(dataframe[target_column].dropna().unique()):
                series = dataframe.loc[
                    dataframe[target_column] == target_value,
                    column,
                ].dropna()

                target_statistics.append(
                    {
                        "target": int(target_value),
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

            results.append(
                {
                    "column": column,
                    "target_statistics": target_statistics,
                }
            )

        return {
            "target_column": target_column,
            "columns_analyzed": len(results),
            "statistics": results,
        }
