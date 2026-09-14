import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from npl_portfolio.ingestion.batch_csv_reader import BatchCSVReader


@dataclass(frozen=True)
class ReferentialRelation:
    """
    Define una relación padre-hijo entre tablas.
    """

    name: str
    child_file: str
    child_key: str

    parent_files: tuple[str, ...]
    parent_key: str


@dataclass(frozen=True)
class ReferentialIntegrityResult:
    """
    Resultado de la validación de una relación.
    """

    relation: str

    child_file: str
    child_key: str

    parent_files: tuple[str, ...]
    parent_key: str

    total_child_rows: int
    orphan_rows: int
    null_key_rows: int
    orphan_percentage: float

    orphan_unique_keys: int
    orphan_samples: list[Any]

    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ReferentialIntegrityValidator:
    """
    Valida integridad referencial entre archivos CSV
    utilizando procesamiento batch.
    """

    def __init__(
        self,
        source_directory: Path,
        output_path: Path,
        reader: BatchCSVReader,
    ) -> None:
        self.source_directory = source_directory
        self.output_path = output_path
        self.reader = reader

        # Evita volver a cargar claves padre
        # que ya fueron utilizadas.
        self._parent_cache: dict[
            tuple[tuple[str, ...], str],
            set[Any],
        ] = {}

    def run(
        self,
        relations: list[ReferentialRelation],
    ) -> list[ReferentialIntegrityResult]:

        print()
        print("INTEGRIDAD REFERENCIAL")
        print("=" * 70)

        results: list[ReferentialIntegrityResult] = []

        for relation in relations:
            result = self._validate_relation(relation)

            results.append(result)

            self._print_result(result)

        self._save_report(results)

        return results

    def _validate_relation(
        self,
        relation: ReferentialRelation,
    ) -> ReferentialIntegrityResult:

        print()
        print(f"Relación: {relation.name}")
        print(f"  {relation.child_file}.{relation.child_key}")
        print("      ↓")
        print(f"  {', '.join(relation.parent_files)}." f"{relation.parent_key}")

        parent_keys = self._load_parent_keys(
            relation.parent_files,
            relation.parent_key,
        )

        child_path = self.source_directory / relation.child_file

        batches, _ = self.reader.read(
            child_path,
            usecols=[relation.child_key],
        )

        total_child_rows = 0
        orphan_rows = 0
        null_key_rows = 0

        orphan_keys: set[Any] = set()

        chunks_processed = 0

        for batch in batches:
            chunks_processed += 1

            total_child_rows += len(batch)

            series = batch[relation.child_key]

            null_mask = series.isna()

            null_count = int(null_mask.sum())

            null_key_rows += null_count

            valid_values = series[~null_mask]

            orphan_mask = ~valid_values.isin(parent_keys)

            batch_orphans = valid_values[orphan_mask]

            orphan_rows += len(batch_orphans)

            orphan_keys.update(batch_orphans.unique().tolist())

            if chunks_processed == 1 or chunks_processed % 10 == 0:
                print(
                    f"    Chunk {chunks_processed:>3} | "
                    f"Filas: {total_child_rows:,} | "
                    f"Huérfanos: {orphan_rows:,}"
                )

            del batch

        invalid_rows = orphan_rows + null_key_rows

        orphan_percentage = (
            (invalid_rows / total_child_rows) * 100 if total_child_rows else 0.0
        )

        samples = list(orphan_keys)[:10]

        passed = orphan_rows == 0 and null_key_rows == 0

        return ReferentialIntegrityResult(
            relation=relation.name,
            child_file=relation.child_file,
            child_key=relation.child_key,
            parent_files=relation.parent_files,
            parent_key=relation.parent_key,
            total_child_rows=total_child_rows,
            orphan_rows=orphan_rows,
            null_key_rows=null_key_rows,
            orphan_percentage=round(
                orphan_percentage,
                6,
            ),
            orphan_unique_keys=len(orphan_keys),
            orphan_samples=samples,
            passed=passed,
        )

    def _load_parent_keys(
        self,
        parent_files: tuple[str, ...],
        parent_key: str,
    ) -> set[Any]:

        cache_key = (
            parent_files,
            parent_key,
        )

        if cache_key in self._parent_cache:
            print("  Reutilizando índice de claves padre.")

            return self._parent_cache[cache_key]

        print("  Construyendo índice de claves padre...")

        parent_keys: set[Any] = set()

        for file_name in parent_files:

            file_path = self.source_directory / file_name

            batches, _ = self.reader.read(
                file_path,
                usecols=[parent_key],
            )

            rows_processed = 0

            for batch in batches:

                rows_processed += len(batch)

                parent_keys.update(batch[parent_key].dropna().unique().tolist())

                del batch

            print(f"    {file_name}: " f"{rows_processed:,} filas")

        print(f"  Claves padre únicas: " f"{len(parent_keys):,}")

        self._parent_cache[cache_key] = parent_keys

        return parent_keys

    def _save_report(
        self,
        results: list[ReferentialIntegrityResult],
    ) -> None:

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        passed = sum(result.passed for result in results)

        failed = len(results) - passed

        payload = {
            "dataset": "Home Credit Default Risk",
            "summary": {
                "relationships": len(results),
                "passed": passed,
                "failed": failed,
            },
            "relationships": [result.to_dict() for result in results],
        }

        with self.output_path.open(
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
        print(f"Reporte generado: " f"{self.output_path}")

    @staticmethod
    def _print_result(
        result: ReferentialIntegrityResult,
    ) -> None:

        status = "PASS" if result.passed else "FAIL"

        print()
        print(f"  Resultado: {status}")
        print(f"  Filas evaluadas: " f"{result.total_child_rows:,}")
        print(f"  Filas huérfanas: " f"{result.orphan_rows:,}")
        print(f"  Claves nulas: " f"{result.null_key_rows:,}")
        print(f"  Claves huérfanas únicas: " f"{result.orphan_unique_keys:,}")
        print(f"  Porcentaje inválido: " f"{result.orphan_percentage:.6f}%")

        if result.orphan_samples:
            print(
                "  Ejemplos: "
                + ", ".join(str(value) for value in result.orphan_samples)
            )

        print("-" * 70)
