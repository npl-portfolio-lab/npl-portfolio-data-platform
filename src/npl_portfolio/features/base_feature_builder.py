from pathlib import Path

import pandas as pd

from npl_portfolio.core.duckdb_manager import DuckDBManager


class BaseDuckDBFeatureBuilder:
    """
    Base para builders de features sobre datasets grandes.

    build():
        Ejecuta la consulta y devuelve únicamente el
        resultado agregado como DataFrame.

    build_to_parquet():
        Ejecuta la misma consulta y escribe directamente
        a Parquet sin materializar el resultado en Pandas.
    """

    def _get_query(self) -> str:
        raise NotImplementedError

    def _get_parameters(self) -> list[str]:
        raise NotImplementedError

    def build(self) -> pd.DataFrame:
        connection = DuckDBManager().connect()

        try:
            return connection.execute(
                self._get_query(),
                self._get_parameters(),
            ).fetchdf()

        finally:
            connection.close()

    def build_to_parquet(
        self,
        output_path: Path,
        overwrite: bool = False,
    ) -> Path:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if output_path.exists() and not overwrite:
            print(
                f"[CHECKPOINT] {output_path.name} "
                "ya existe. Se omite."
            )
            return output_path

        connection = DuckDBManager().connect()

        try:
            escaped_output = str(
                output_path.resolve()
            ).replace("'", "''")

            query = self._get_query()

            copy_query = f"""
                COPY (
                    {query}
                )
                TO '{escaped_output}'
                (
                    FORMAT PARQUET,
                    COMPRESSION ZSTD
                )
            """

            print(
                f"[OUT-OF-CORE] Generando "
                f"{output_path.name}..."
            )

            connection.execute(
                copy_query,
                self._get_parameters(),
            )

        finally:
            connection.close()

        print(
            f"[OK] Checkpoint creado: "
            f"{output_path}"
        )

        return output_path
