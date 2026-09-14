import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from npl_portfolio.ingestion.batch_csv_reader import BatchCSVReader


@dataclass(frozen=True)
class ReferentialDiagnosticResult:
    relation: str
    child_file: str

    total_rows: int

    missing_parent_rows: int
    missing_parent_unique_prev: int

    missing_parent_but_valid_curr_rows: int
    fully_disconnected_rows: int

    existing_prev_rows: int
    mismatched_curr_rows: int

    mismatch_percentage: float
    disconnected_percentage: float

    missing_parent_samples: list[Any]
    mismatch_samples: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ReferentialDiagnostics:
    """
    Analiza en detalle las relaciones que fallaron
    durante la validación de integridad referencial.
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

        self.application_curr_ids: set[Any] = set()
        self.previous_map: dict[Any, Any] = {}

    def run(
        self,
        child_files: list[str],
    ) -> list[ReferentialDiagnosticResult]:

        print()
        print("DIAGNÓSTICO DE INTEGRIDAD REFERENCIAL")
        print("=" * 75)

        self._load_application_curr_ids()
        self._load_previous_application_map()

        results: list[ReferentialDiagnosticResult] = []

        for child_file in child_files:
            result = self._diagnose_child_file(child_file)

            results.append(result)

            self._print_result(result)

        self._save_report(results)

        return results

    def _load_application_curr_ids(
        self,
    ) -> None:

        print()
        print("Construyendo índice global de SK_ID_CURR...")

        application_files = (
            "application_train.csv",
            "application_test.csv",
        )

        for file_name in application_files:

            file_path = self.source_directory / file_name

            batches, _ = self.reader.read(
                file_path,
                usecols=["SK_ID_CURR"],
            )

            rows_processed = 0

            for batch in batches:

                rows_processed += len(batch)

                self.application_curr_ids.update(
                    batch["SK_ID_CURR"].dropna().unique().tolist()
                )

                del batch

            print(f"  {file_name}: " f"{rows_processed:,} filas")

        print(f"SK_ID_CURR únicos: " f"{len(self.application_curr_ids):,}")

    def _load_previous_application_map(
        self,
    ) -> None:

        print()
        print("Construyendo mapa " "SK_ID_PREV → SK_ID_CURR...")

        file_path = self.source_directory / "previous_application.csv"

        batches, _ = self.reader.read(
            file_path,
            usecols=[
                "SK_ID_PREV",
                "SK_ID_CURR",
            ],
        )

        rows_processed = 0

        for batch in batches:

            rows_processed += len(batch)

            clean_batch = batch.dropna(
                subset=[
                    "SK_ID_PREV",
                    "SK_ID_CURR",
                ]
            )

            self.previous_map.update(
                zip(
                    clean_batch["SK_ID_PREV"].tolist(),
                    clean_batch["SK_ID_CURR"].tolist(),
                )
            )

            del batch
            del clean_batch

        print(f"previous_application.csv: " f"{rows_processed:,} filas")

        print(f"Relaciones SK_ID_PREV únicas: " f"{len(self.previous_map):,}")

    def _diagnose_child_file(
        self,
        child_file: str,
    ) -> ReferentialDiagnosticResult:

        print()
        print("-" * 75)
        print(f"Analizando: {child_file}")

        file_path = self.source_directory / child_file

        batches, _ = self.reader.read(
            file_path,
            usecols=[
                "SK_ID_PREV",
                "SK_ID_CURR",
            ],
        )

        total_rows = 0

        missing_parent_rows = 0
        missing_parent_but_valid_curr_rows = 0
        fully_disconnected_rows = 0

        existing_prev_rows = 0
        mismatched_curr_rows = 0

        missing_prev_keys: set[Any] = set()

        mismatch_samples: list[dict[str, Any]] = []

        chunks_processed = 0

        for batch in batches:

            chunks_processed += 1
            total_rows += len(batch)

            prev_series = batch["SK_ID_PREV"]

            curr_series = batch["SK_ID_CURR"]

            parent_curr_series = prev_series.map(self.previous_map)

            missing_parent_mask = parent_curr_series.isna()

            missing_batch = batch[missing_parent_mask]

            missing_count = len(missing_batch)

            missing_parent_rows += missing_count

            if missing_count:

                missing_prev_keys.update(
                    missing_batch["SK_ID_PREV"].dropna().unique().tolist()
                )

                valid_curr_mask = missing_batch["SK_ID_CURR"].isin(
                    self.application_curr_ids
                )

                valid_curr_count = int(valid_curr_mask.sum())

                missing_parent_but_valid_curr_rows += valid_curr_count

                fully_disconnected_rows += missing_count - valid_curr_count

            existing_mask = ~missing_parent_mask

            existing_prev_rows += int(existing_mask.sum())

            if existing_mask.any():

                child_curr_existing = curr_series[existing_mask]

                expected_curr = parent_curr_series[existing_mask]

                mismatch_mask = child_curr_existing.values != expected_curr.values

                mismatch_count = int(mismatch_mask.sum())

                mismatched_curr_rows += mismatch_count

                if mismatch_count > 0 and len(mismatch_samples) < 10:

                    mismatch_positions = pd.Series(
                        mismatch_mask,
                        index=child_curr_existing.index,
                    )

                    mismatch_indexes = mismatch_positions[
                        mismatch_positions
                    ].index.tolist()

                    for index in mismatch_indexes:

                        if len(mismatch_samples) >= 10:
                            break

                        mismatch_samples.append(
                            {
                                "SK_ID_PREV": (
                                    batch.at[
                                        index,
                                        "SK_ID_PREV",
                                    ]
                                ),
                                "child_SK_ID_CURR": (
                                    batch.at[
                                        index,
                                        "SK_ID_CURR",
                                    ]
                                ),
                                "expected_SK_ID_CURR": (
                                    self.previous_map.get(
                                        batch.at[
                                            index,
                                            "SK_ID_PREV",
                                        ]
                                    )
                                ),
                            }
                        )

            if chunks_processed == 1 or chunks_processed % 10 == 0:
                print(
                    f"  Chunk "
                    f"{chunks_processed:>3} | "
                    f"Filas: "
                    f"{total_rows:,} | "
                    f"Sin SK_ID_PREV padre: "
                    f"{missing_parent_rows:,} | "
                    f"Mismatch: "
                    f"{mismatched_curr_rows:,}"
                )

            del batch

        mismatch_percentage = (
            (mismatched_curr_rows / existing_prev_rows) * 100
            if existing_prev_rows
            else 0.0
        )

        disconnected_percentage = (
            (fully_disconnected_rows / total_rows) * 100 if total_rows else 0.0
        )

        return ReferentialDiagnosticResult(
            relation=(f"{child_file}" "_to_previous_application"),
            child_file=child_file,
            total_rows=total_rows,
            missing_parent_rows=(missing_parent_rows),
            missing_parent_unique_prev=len(missing_prev_keys),
            missing_parent_but_valid_curr_rows=(missing_parent_but_valid_curr_rows),
            fully_disconnected_rows=(fully_disconnected_rows),
            existing_prev_rows=(existing_prev_rows),
            mismatched_curr_rows=(mismatched_curr_rows),
            mismatch_percentage=round(
                mismatch_percentage,
                6,
            ),
            disconnected_percentage=round(
                disconnected_percentage,
                6,
            ),
            missing_parent_samples=list(missing_prev_keys)[:10],
            mismatch_samples=(mismatch_samples),
        )

    def _save_report(
        self,
        results: list[ReferentialDiagnosticResult],
    ) -> None:

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "dataset": ("Home Credit Default Risk"),
            "diagnostics": [result.to_dict() for result in results],
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
        result: ReferentialDiagnosticResult,
    ) -> None:

        print()
        print(f"Resultado: " f"{result.child_file}")

        print(f"  Filas evaluadas: " f"{result.total_rows:,}")

        print(f"  Sin SK_ID_PREV padre: " f"{result.missing_parent_rows:,}")

        print(
            f"  SK_ID_PREV huérfanos únicos: " f"{result.missing_parent_unique_prev:,}"
        )

        print(
            "  Sin padre PREV pero con "
            "SK_ID_CURR válido: "
            f"{result.missing_parent_but_valid_curr_rows:,}"
        )

        print(f"  Totalmente desconectadas: " f"{result.fully_disconnected_rows:,}")

        print(f"  Filas con SK_ID_PREV existente: " f"{result.existing_prev_rows:,}")

        print(f"  SK_ID_CURR inconsistente: " f"{result.mismatched_curr_rows:,}")

        print(
            f"  % mismatch sobre PREV existentes: " f"{result.mismatch_percentage:.6f}%"
        )

        print(f"  % totalmente desconectado: " f"{result.disconnected_percentage:.6f}%")
