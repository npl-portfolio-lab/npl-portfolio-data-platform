import json
from pathlib import Path

from npl_portfolio.transformation.home_credit_transformer import (
    HomeCreditTransformer,
)


class TransformationPipeline:
    """
    Orquesta la transformación de los archivos
    Home Credit desde CSV hacia Parquet.
    """

    def __init__(
        self,
        transformer: HomeCreditTransformer,
        source_directory: Path,
        output_directory: Path,
        report_path: Path,
    ) -> None:
        self.transformer = transformer
        self.source_directory = source_directory
        self.output_directory = output_directory
        self.report_path = report_path

    def run(
        self,
        files: list[str],
    ) -> None:

        results = []

        print()
        print("TRANSFORMACIÓN HOME CREDIT")
        print("=" * 75)

        for file_name in files:

            source_path = self.source_directory / file_name

            output_name = Path(file_name).with_suffix(".parquet").name

            output_path = self.output_directory / output_name

            result = self.transformer.transform(
                source_path=source_path,
                output_path=output_path,
            )

            results.append(result.to_dict())

            print(f"  CSV: {result.source_size_mb:.2f} MB")

            print(f"  Parquet: " f"{result.output_size_mb:.2f} MB")

            print(f"  Compresión: " f"{result.compression_ratio:.2f}x")

            print(f"  Filas: " f"{result.row_count:,}")

        self._save_report(results)

    def _save_report(
        self,
        results: list[dict],
    ) -> None:

        self.report_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "dataset": ("Home Credit Default Risk"),
            "format": "parquet",
            "tables_transformed": len(results),
            "tables": results,
        }

        with self.report_path.open(
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
        print(f"Reporte generado: " f"{self.report_path}")
