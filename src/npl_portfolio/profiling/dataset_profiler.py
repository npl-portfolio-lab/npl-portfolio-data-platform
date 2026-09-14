import gc
import json
from pathlib import Path

from npl_portfolio.profiling.csv_profiler import CSVDataProfiler
from npl_portfolio.profiling.models import TableProfile


class DatasetProfiler:
    """Orquesta el perfilado de todos los archivos CSV de un dataset."""

    def __init__(
        self,
        source_directory: Path,
        output_directory: Path,
        csv_profiler: CSVDataProfiler,
    ) -> None:
        self.source_directory = source_directory
        self.output_directory = output_directory
        self.csv_profiler = csv_profiler

    def run(self) -> list[TableProfile]:
        csv_files = sorted(self.source_directory.glob("*.csv"))

        if not csv_files:
            raise FileNotFoundError(
                f"No se encontraron archivos CSV en {self.source_directory}"
            )

        print()
        print("PERFILADO DEL DATASET")
        print("=" * 60)
        print(f"Archivos encontrados: {len(csv_files)}")
        print()

        profiles: list[TableProfile] = []

        for file_path in csv_files:
            profile = self.csv_profiler.profile(file_path)

            profiles.append(profile)

            self._print_summary(profile)

            gc.collect()

        self._save_report(profiles)

        return profiles

    def _save_report(
        self,
        profiles: list[TableProfile],
    ) -> None:
        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = self.output_directory / "home_credit_profile.json"

        payload = {
            "dataset": "Home Credit Default Risk",
            "tables": [profile.to_dict() for profile in profiles],
        }

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                payload,
                file,
                indent=2,
                ensure_ascii=False,
            )

        print()
        print(f"Reporte generado: {output_path}")

    @staticmethod
    def _print_summary(
        profile: TableProfile,
    ) -> None:
        print(f"  Encoding: {profile.encoding}")
        print(f"  Chunks: {profile.chunks_processed:,}")
        print(f"  Filas: {profile.row_count:,}")
        print(f"  Columnas: {profile.column_count:,}")
        print(f"  Duplicados detectados: " f"{profile.duplicate_rows:,}")
        print(f"  Memoria procesada: " f"{profile.memory_usage_mb:,.2f} MB")

        if profile.detected_keys:
            print("  Claves: " + ", ".join(profile.detected_keys))
        else:
            print("  Claves: Ninguna")

        print("-" * 60)
