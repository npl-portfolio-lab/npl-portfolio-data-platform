from pathlib import Path
from typing import Any

from npl_portfolio.ingestion.batch_csv_reader import BatchCSVReader
from npl_portfolio.profiling.models import ColumnProfile, TableProfile


class CSVDataProfiler:
    """Perfilador CSV basado en procesamiento batch."""

    KNOWN_KEYS = {
        "SK_ID_CURR",
        "SK_ID_PREV",
        "SK_ID_BUREAU",
    }

    MAX_UNIQUE_VALUES = 100_000

    def __init__(
        self,
        reader: BatchCSVReader,
    ) -> None:
        self.reader = reader

    def profile(
        self,
        file_path: Path,
    ) -> TableProfile:
        """
        Perfila un archivo CSV utilizando lectura por lotes.

        No carga el archivo completo en memoria.
        """

        print(f"Perfilando: {file_path.name}")

        batches, encoding = self.reader.read(file_path)

        total_rows = 0
        total_memory_bytes = 0
        total_duplicates = 0
        chunks_processed = 0

        columns: list[str] = []

        column_stats: dict[str, dict[str, int]] = {}
        column_types: dict[str, str] = {}

        detected_keys: set[str] = set()

        key_values: dict[str, set[Any]] = {key: set() for key in self.KNOWN_KEYS}

        unique_values: dict[str, set[Any]] = {}

        for batch in batches:
            chunks_processed += 1

            batch_rows = len(batch)
            total_rows += batch_rows

            if not columns:
                columns = list(batch.columns)

                for column in columns:
                    column_stats[column] = {
                        "null_count": 0,
                    }

                    unique_values[column] = set()

            total_memory_bytes += int(
                batch.memory_usage(
                    index=True,
                    deep=True,
                ).sum()
            )

            # Duplicados detectados dentro del chunk.
            total_duplicates += int(batch.duplicated().sum())

            for column in batch.columns:
                series = batch[column]

                column_types[column] = str(series.dtype)

                column_stats[column]["null_count"] += int(series.isna().sum())

                if column in self.KNOWN_KEYS:
                    detected_keys.add(column)

                    key_values[column].update(series.dropna().tolist())

                else:
                    current_values = unique_values[column]

                    if len(current_values) < self.MAX_UNIQUE_VALUES:
                        remaining_capacity = self.MAX_UNIQUE_VALUES - len(
                            current_values
                        )

                        new_values = (
                            series.dropna()
                            .drop_duplicates()
                            .head(remaining_capacity)
                            .tolist()
                        )

                        current_values.update(new_values)

            print(
                f"  Chunk {chunks_processed:>3} | " f"Filas acumuladas: {total_rows:,}"
            )

            del batch

        column_profiles: list[ColumnProfile] = []

        for column in columns:
            null_count = column_stats[column]["null_count"]

            null_percentage = (null_count / total_rows) * 100 if total_rows else 0.0

            if column in self.KNOWN_KEYS:
                unique_count = len(key_values[column])
            else:
                unique_count = len(unique_values[column])

            column_profiles.append(
                ColumnProfile(
                    name=column,
                    dtype=column_types[column],
                    null_count=null_count,
                    null_percentage=round(
                        null_percentage,
                        2,
                    ),
                    unique_count=unique_count,
                )
            )

        return TableProfile(
            file_name=file_path.name,
            file_size_mb=round(
                file_path.stat().st_size / (1024**2),
                2,
            ),
            row_count=total_rows,
            column_count=len(columns),
            duplicate_rows=total_duplicates,
            memory_usage_mb=round(
                total_memory_bytes / (1024**2),
                2,
            ),
            detected_keys=sorted(detected_keys),
            columns=column_profiles,
            encoding=encoding,
            chunks_processed=chunks_processed,
        )
