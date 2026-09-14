from pathlib import Path

from npl_portfolio.data_quality.home_credit_value_rules import (
    build_home_credit_value_rules,
)
from npl_portfolio.data_quality.value_quality_engine import (
    ValueQualityEngine,
)
from npl_portfolio.ingestion.batch_csv_reader import (
    BatchCSVReader,
)


def main() -> None:

    project_root = Path(__file__).resolve().parents[1]

    source_directory = project_root / "data" / "raw" / "home_credit"

    output_path = (
        project_root / "data" / "interim" / "quality" / "value_quality_report.json"
    )

    reader = BatchCSVReader(
        chunk_size=100_000,
    )

    engine = ValueQualityEngine(
        source_directory=source_directory,
        output_path=output_path,
        reader=reader,
    )

    results = engine.run(build_home_credit_value_rules())

    passed = sum(result.passed for result in results)

    failed = len(results) - passed

    print()
    print("=" * 75)
    print("CALIDAD DE VALORES FINALIZADA")
    print(f"Validaciones: {len(results)}")
    print(f"PASS: {passed}")
    print(f"FAIL: {failed}")
    print("=" * 75)


if __name__ == "__main__":
    main()
