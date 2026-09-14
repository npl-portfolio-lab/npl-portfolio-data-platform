from pathlib import Path

from npl_portfolio.data_quality.engine import DataQualityEngine
from npl_portfolio.data_quality.home_credit_rules import (
    build_home_credit_rules,
)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    profile_path = (
        project_root / "data" / "interim" / "profiling" / "home_credit_profile.json"
    )

    output_path = (
        project_root
        / "data"
        / "interim"
        / "quality"
        / "home_credit_quality_report.json"
    )

    engine = DataQualityEngine(
        profile_path=profile_path,
        output_path=output_path,
        rules=build_home_credit_rules(),
    )

    report = engine.run()

    print()
    print("=" * 60)
    print("DATA QUALITY FINALIZADO")
    print(f"Checks: {report.total_checks}")
    print(f"PASS: {report.passed_checks}")
    print(f"FAIL: {report.failed_checks}")
    print("Estado general: " + ("APROBADO" if report.passed else "CON INCIDENCIAS"))
    print("=" * 60)


if __name__ == "__main__":
    main()
