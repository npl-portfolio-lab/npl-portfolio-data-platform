from collections.abc import Iterator
from pathlib import Path

import pandas as pd


class BatchCSVReader:
    """
    Lee archivos CSV por lotes y detecta su codificación
    antes de iniciar el procesamiento.
    """

    ENCODINGS = (
        "utf-8",
        "cp1252",
        "latin1",
    )

    ENCODING_SAMPLE_SIZE = 1_048_576  # 1 MB

    def __init__(
        self,
        chunk_size: int = 100_000,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size debe ser mayor que cero.")

        self.chunk_size = chunk_size

    def read(
        self,
        file_path: Path,
    ) -> tuple[Iterator[pd.DataFrame], str]:
        """
        Detecta el encoding y devuelve un iterador de DataFrames.

        El CSV nunca se carga completamente en memoria.
        """

        self._validate_file(file_path)

        encoding = self._detect_encoding(file_path)

        iterator = pd.read_csv(
            file_path,
            encoding=encoding,
            chunksize=self.chunk_size,
            low_memory=False,
        )

        return iterator, encoding

    def _detect_encoding(
        self,
        file_path: Path,
    ) -> str:
        """
        Detecta la codificación leyendo una muestra binaria
        mucho mayor que unas pocas filas.

        Para archivos menores a 1 MB analiza el archivo completo.
        """

        with file_path.open("rb") as file:
            sample = file.read(self.ENCODING_SAMPLE_SIZE)

        for encoding in self.ENCODINGS:
            try:
                sample.decode(
                    encoding,
                    errors="strict",
                )

                return encoding

            except UnicodeDecodeError:
                continue

        raise UnicodeError(
            f"No fue posible detectar una codificación válida " f"para {file_path.name}"
        )

    @staticmethod
    def _validate_file(
        file_path: Path,
    ) -> None:
        if not file_path.exists():
            raise FileNotFoundError(f"No existe el archivo: {file_path}")

        if not file_path.is_file():
            raise ValueError(f"La ruta no corresponde a un archivo: {file_path}")

        if file_path.suffix.lower() != ".csv":
            raise ValueError(f"El archivo no es CSV: {file_path}")
