from pathlib import Path
import subprocess
import zipfile


class HomeCreditDownloader:
    """
    Descarga y extrae el dataset Home Credit Default Risk desde Kaggle.

    La clase utiliza Kaggle CLI, cuya autenticación debe estar
    configurada previamente en el entorno.
    """

    COMPETITION = "home-credit-default-risk"

    def __init__(self, destination: Path) -> None:
        self.destination = destination

    def download(self) -> None:
        """
        Descarga el dataset completo desde Kaggle.
        """

        self.destination.mkdir(parents=True, exist_ok=True)

        print("Descargando Home Credit Default Risk...")

        command = [
            "kaggle",
            "competitions",
            "download",
            "-c",
            self.COMPETITION,
            "-p",
            str(self.destination),
        ]

        subprocess.run(
            command,
            check=True,
        )

        print("Descarga finalizada.")

    def extract(self) -> None:
        """
        Extrae los archivos ZIP descargados.
        """

        zip_files = list(self.destination.glob("*.zip"))

        if not zip_files:
            raise FileNotFoundError(
                f"No se encontró ningún archivo ZIP en {self.destination}"
            )

        for zip_path in zip_files:
            print(f"Extrayendo: {zip_path.name}")

            with zipfile.ZipFile(zip_path, "r") as zip_file:
                zip_file.extractall(self.destination)

        print("Extracción finalizada.")

    def cleanup(self) -> None:
        """
        Elimina los ZIP después de la extracción.
        """

        for zip_path in self.destination.glob("*.zip"):
            print(f"Eliminando archivo comprimido: {zip_path.name}")
            zip_path.unlink()

    def run(self) -> None:
        """
        Ejecuta el flujo completo de adquisición.
        """

        self.download()
        self.extract()
        self.cleanup()

        print()
        print("Dataset Home Credit disponible en:")
        print(self.destination)
