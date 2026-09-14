from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


class ParquetWriter:
    """
    Escribe DataFrames en formato Parquet de forma incremental.
    """

    def __init__(
        self,
        compression: str = "snappy",
    ) -> None:
        self.compression = compression

    def write_batches(
        self,
        batches,
        output_path: Path,
    ) -> tuple[int, int]:
        """
        Escribe múltiples batches en un único archivo Parquet.

        Retorna:
            total_rows
            total_batches
        """

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        writer: pq.ParquetWriter | None = None

        total_rows = 0
        total_batches = 0

        try:
            for batch in batches:

                total_batches += 1
                total_rows += len(batch)

                table = pa.Table.from_pandas(
                    batch,
                    preserve_index=False,
                )

                if writer is None:
                    writer = pq.ParquetWriter(
                        output_path,
                        table.schema,
                        compression=self.compression,
                    )

                writer.write_table(table)

                print(
                    f"  Batch {total_batches:>3} | " f"Filas escritas: {total_rows:,}"
                )

                del batch
                del table

        finally:
            if writer is not None:
                writer.close()

        return total_rows, total_batches
