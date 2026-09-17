from pathlib import Path

import duckdb


class DuckDBManager:
    """
    Configuración central de DuckDB para procesamiento
    analítico out-of-core.

    Objetivos:
    - Limitar memoria administrada por DuckDB.
    - Permitir spill a disco.
    - Limitar paralelismo.
    - Evitar configuraciones distintas entre pipelines.
    """

    DEFAULT_MEMORY_LIMIT = "2GB"
    DEFAULT_THREADS = 4

    def __init__(
        self,
        memory_limit: str = DEFAULT_MEMORY_LIMIT,
        threads: int = DEFAULT_THREADS,
    ) -> None:
        self.memory_limit = memory_limit
        self.threads = threads

        self.project_root = Path(__file__).resolve().parents[3]

        self.temp_directory = self.project_root / "data" / "interim" / "duckdb_tmp"

        self.temp_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    def connect(self) -> duckdb.DuckDBPyConnection:
        connection = duckdb.connect()

        connection.execute(f"SET memory_limit = '{self.memory_limit}'")

        connection.execute(f"SET threads = {self.threads}")

        temp_path = str(self.temp_directory.resolve()).replace("'", "''")

        connection.execute(f"SET temp_directory = '{temp_path}'")

        return connection
