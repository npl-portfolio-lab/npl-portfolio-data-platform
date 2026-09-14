from pathlib import Path

from npl_portfolio.ingestion.batch_csv_reader import (
    BatchCSVReader,
)
from npl_portfolio.pipelines.transformation_pipeline import (
    TransformationPipeline,
)
from npl_portfolio.transformation.home_credit_transformer import (
    HomeCreditTransformer,
)
from npl_portfolio.transformation.parquet_writer import (
    ParquetWriter,
)


def main() -> None:

    project_root = Path(__file__).resolve().parents[1]

    source_directory = project_root / "data" / "raw" / "home_credit"

    output_directory = project_root / "data" / "processed" / "home_credit"

    report_path = (
        project_root
        / "data"
        / "interim"
        / "transformation"
        / "home_credit_transformation_report.json"
    )

    files = [
        "application_train.csv",
        "application_test.csv",
        "bureau.csv",
        "bureau_balance.csv",
        "previous_application.csv",
        "POS_CASH_balance.csv",
        "credit_card_balance.csv",
        "installments_payments.csv",
    ]

    reader = BatchCSVReader(
        chunk_size=100_000,
    )

    writer = ParquetWriter(
        compression="snappy",
    )

    transformer = HomeCreditTransformer(
        reader=reader,
        writer=writer,
    )

    pipeline = TransformationPipeline(
        transformer=transformer,
        source_directory=source_directory,
        output_directory=output_directory,
        report_path=report_path,
    )

    pipeline.run(files)


if __name__ == "__main__":
    main()
