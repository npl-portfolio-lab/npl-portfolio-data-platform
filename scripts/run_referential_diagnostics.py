from pathlib import Path

from npl_portfolio.data_quality.referential_diagnostics import (
    ReferentialDiagnostics,
)
from npl_portfolio.ingestion.batch_csv_reader import (
    BatchCSVReader,
)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    source_directory = project_root / "data" / "raw" / "home_credit"

    output_path = (
        project_root
        / "data"
        / "interim"
        / "quality"
        / "referential_diagnostics_report.json"
    )

    reader = BatchCSVReader(
        chunk_size=100_000,
    )

    diagnostics = ReferentialDiagnostics(
        source_directory=source_directory,
        output_path=output_path,
        reader=reader,
    )

    child_files = [
        "POS_CASH_balance.csv",
        "credit_card_balance.csv",
        "installments_payments.csv",
    ]

    results = diagnostics.run(child_files)

    print()
    print("=" * 75)
    print("DIAGNÓSTICO REFERENCIAL FINALIZADO")
    print(f"Tablas analizadas: " f"{len(results)}")
    print("=" * 75)


if __name__ == "__main__":
    main()
