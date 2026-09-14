import json
from pathlib import Path
from typing import Any

from npl_portfolio.data_quality.models import QualityReport
from npl_portfolio.data_quality.rules import DataQualityRule


class DataQualityEngine:
    """Ejecuta reglas de calidad contra un perfil de dataset."""

    def __init__(
        self,
        profile_path: Path,
        output_path: Path,
        rules: dict[str, list[DataQualityRule]],
    ) -> None:
        self.profile_path = profile_path
        self.output_path = output_path
        self.rules = rules

    def run(self) -> QualityReport:
        profile = self._load_profile()

        table_index = {
            table["file_name"]: table
            for table in profile["tables"]
        }

        report = QualityReport(
            dataset=profile["dataset"],
        )

        print()
        print("DATA QUALITY")
        print("=" * 60)

        for table_name, rules in self.rules.items():

            print()
            print(f"Tabla: {table_name}")

            table_profile = table_index.get(
                table_name
            )

            if table_profile is None:
                print("  ERROR: tabla no encontrada")
                continue

            for rule in rules:
                result = rule.validate(
                    table_name,
                    table_profile,
                )

                report.checks.append(result)

                status = (
                    "PASS"
                    if result.passed
                    else "FAIL"
                )

                print(
                    f"  [{status}] "
                    f"{result.rule}"
                    + (
                        f" -> {result.column}"
                        if result.column
                        else ""
                    )
                )

        self._save_report(report)

        return report

    def _load_profile(
        self,
    ) -> dict[str, Any]:

        if not self.profile_path.exists():
            raise FileNotFoundError(
                f"No existe el perfil: "
                f"{self.profile_path}"
            )

        with self.profile_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    def _save_report(
        self,
        report: QualityReport,
    ) -> None:

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self.output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                report.to_dict(),
                file,
                indent=2,
                ensure_ascii=False,
            )

        print()
        print(
            f"Reporte generado: "
            f"{self.output_path}"
        )