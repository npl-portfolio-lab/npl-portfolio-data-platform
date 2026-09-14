from pathlib import Path

from npl_portfolio.ingestion.batch_csv_reader import (
    BatchCSVReader,
)
from npl_portfolio.transformation.models import (
    TransformationResult,
)
from npl_portfolio.transformation.parquet_writer import (
    ParquetWriter,
)


class HomeCreditTransformer:
    """
    Transforma los CSV del dataset Home Credit
    hacia archivos Parquet optimizados.
    """

    def __init__(
        self,
        reader: BatchCSVReader,
        writer: ParquetWriter,
    ) -> None:
        self.reader = reader
        self.writer = writer

    def transform(
        self,
        source_path: Path,
        output_path: Path,
    ) -> TransformationResult:

        print()
        print(f"Transformando: " f"{source_path.name}")

        batches, _ = self.reader.read(source_path)

        row_count, batch_count = self.writer.write_batches(
            batches=batches,
            output_path=output_path,
        )

        source_size_mb = source_path.stat().st_size / (1024**2)

        output_size_mb = output_path.stat().st_size / (1024**2)

        compression_ratio = (
            source_size_mb / output_size_mb if output_size_mb > 0 else 0.0
        )

        return TransformationResult(
            source_file=source_path.name,
            output_file=output_path.name,
            source_size_mb=round(
                source_size_mb,
                2,
            ),
            output_size_mb=round(
                output_size_mb,
                2,
            ),
            row_count=row_count,
            batch_count=batch_count,
            compression_ratio=round(
                compression_ratio,
                2,
            ),
        )
