import json
from pathlib import Path
from typing import Any

from npl_portfolio.data_quality.value_models import (
    ValueQualityResult,
)
from npl_portfolio.data_quality.value_rules import (
    ValueQualityRule,
)
from npl_portfolio.ingestion.batch_csv_reader import (
    BatchCSVReader,
)


class ValueQualityEngine:
    """
    Ejecuta reglas de calidad semántica
    sobre los CSV utilizando procesamiento batch.
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

    def run(
        self,
        rules_by_table: dict[
            str,
            list[ValueQualityRule],
        ],
    ) -> list[ValueQualityResult]:

        print()
        print("CALIDAD SEMÁNTICA Y DE VALORES")
        print("=" * 75)

        results: list[ValueQualityResult] = []

        for table, rules in rules_by_table.items():

            table_results = self._validate_table(
                table=table,
                rules=rules,
            )

            results.extend(table_results)

        self._save_report(results)

        return results

    def _validate_table(
        self,
        table: str,
        rules: list[ValueQualityRule],
    ) -> list[ValueQualityResult]:

        print()
        print(f"Tabla: {table}")

        file_path = self.source_directory / table

        required_columns = sorted({rule.column for rule in rules})

        batches, _ = self.reader.read(
            file_path,
            usecols=required_columns,
        )

        stats: dict[str, dict[str, Any]] = {}

        for rule in rules:

            key = self._rule_key(rule)

            stats[key] = {
                "rule": rule,
                "rows_evaluated": 0,
                "null_rows": 0,
                "invalid_rows": 0,
                "invalid_samples": [],
            }

        chunks_processed = 0
        rows_processed = 0

        for batch in batches:

            chunks_processed += 1
            rows_processed += len(batch)

            for rule in rules:

                key = self._rule_key(rule)
                rule_stats = stats[key]

                series = batch[rule.column]

                rule_stats["rows_evaluated"] += len(series)

                rule_stats["null_rows"] += int(series.isna().sum())

                invalid_mask = rule.invalid_mask(series)

                invalid_values = series[invalid_mask]

                rule_stats["invalid_rows"] += len(invalid_values)

                samples = rule_stats["invalid_samples"]

                if len(samples) < 10 and not invalid_values.empty:

                    remaining = 10 - len(samples)

                    new_samples = (
                        invalid_values.drop_duplicates().head(remaining).tolist()
                    )

                    samples.extend(new_samples)

            if chunks_processed == 1 or chunks_processed % 10 == 0:
                print(
                    f"  Chunk "
                    f"{chunks_processed:>3} | "
                    f"Filas acumuladas: "
                    f"{rows_processed:,}"
                )

            del batch

        results: list[ValueQualityResult] = []

        for rule in rules:

            key = self._rule_key(rule)
            rule_stats = stats[key]

            rows_evaluated = rule_stats["rows_evaluated"]

            invalid_rows = rule_stats["invalid_rows"]

            invalid_percentage = (
                (invalid_rows / rows_evaluated) * 100 if rows_evaluated else 0.0
            )

            result = ValueQualityResult(
                table=table,
                column=rule.column,
                rule=rule.name,
                rows_evaluated=(rows_evaluated),
                null_rows=(rule_stats["null_rows"]),
                invalid_rows=(invalid_rows),
                invalid_percentage=round(
                    invalid_percentage,
                    6,
                ),
                invalid_samples=(rule_stats["invalid_samples"]),
                passed=(invalid_rows == 0),
            )

            results.append(result)

            self._print_result(result)

        return results

    @staticmethod
    def _rule_key(
        rule: ValueQualityRule,
    ) -> str:

        return f"{rule.name}:" f"{rule.column}"

    def _save_report(
        self,
        results: list[ValueQualityResult],
    ) -> None:

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        passed = sum(result.passed for result in results)

        failed = len(results) - passed

        payload = {
            "dataset": ("Home Credit Default Risk"),
            "summary": {
                "total_checks": len(results),
                "passed": passed,
                "failed": failed,
            },
            "checks": [result.to_dict() for result in results],
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
                default=self._json_safe,
            )

        print()
        print(f"Reporte generado: " f"{self.output_path}")

    @staticmethod
    def _json_safe(
        value: Any,
    ) -> Any:

        if hasattr(value, "item"):
            return value.item()

        return str(value)

    @staticmethod
    def _print_result(
        result: ValueQualityResult,
    ) -> None:

        status = "PASS" if result.passed else "FAIL"

        print(
            f"  [{status}] "
            f"{result.column} | "
            f"{result.rule} | "
            f"Inválidos: "
            f"{result.invalid_rows:,} "
            f"({result.invalid_percentage:.6f}%)"
        )

        if result.invalid_samples:

            print(
                "         Ejemplos: "
                + ", ".join(str(value) for value in result.invalid_samples)
            )
