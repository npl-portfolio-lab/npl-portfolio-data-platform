from pathlib import Path

from npl_portfolio.data_quality.referential_integrity import (
    ReferentialIntegrityValidator,
    ReferentialRelation,
)
from npl_portfolio.ingestion.batch_csv_reader import BatchCSVReader


def build_relationships() -> list[ReferentialRelation]:
    return [
        ReferentialRelation(
            name="bureau_to_application",
            child_file="bureau.csv",
            child_key="SK_ID_CURR",
            parent_files=(
                "application_train.csv",
                "application_test.csv",
            ),
            parent_key="SK_ID_CURR",
        ),

        ReferentialRelation(
            name="bureau_balance_to_bureau",
            child_file="bureau_balance.csv",
            child_key="SK_ID_BUREAU",
            parent_files=(
                "bureau.csv",
            ),
            parent_key="SK_ID_BUREAU",
        ),

        ReferentialRelation(
            name="previous_application_to_application",
            child_file="previous_application.csv",
            child_key="SK_ID_CURR",
            parent_files=(
                "application_train.csv",
                "application_test.csv",
            ),
            parent_key="SK_ID_CURR",
        ),

        ReferentialRelation(
            name="pos_cash_to_previous_application",
            child_file="POS_CASH_balance.csv",
            child_key="SK_ID_PREV",
            parent_files=(
                "previous_application.csv",
            ),
            parent_key="SK_ID_PREV",
        ),

        ReferentialRelation(
            name="credit_card_to_previous_application",
            child_file="credit_card_balance.csv",
            child_key="SK_ID_PREV",
            parent_files=(
                "previous_application.csv",
            ),
            parent_key="SK_ID_PREV",
        ),

        ReferentialRelation(
            name="installments_to_previous_application",
            child_file="installments_payments.csv",
            child_key="SK_ID_PREV",
            parent_files=(
                "previous_application.csv",
            ),
            parent_key="SK_ID_PREV",
        ),
    ]


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    source_directory = (
        project_root
        / "data"
        / "raw"
        / "home_credit"
    )

    output_path = (
        project_root
        / "data"
        / "interim"
        / "quality"
        / "referential_integrity_report.json"
    )

    reader = BatchCSVReader(
        chunk_size=100_000,
    )

    validator = ReferentialIntegrityValidator(
        source_directory=source_directory,
        output_path=output_path,
        reader=reader,
    )

    results = validator.run(
        build_relationships()
    )

    passed = sum(
        result.passed
        for result in results
    )

    failed = (
        len(results)
        - passed
    )

    print()
    print("=" * 70)
    print("INTEGRIDAD REFERENCIAL FINALIZADA")
    print(f"Relaciones: {len(results)}")
    print(f"PASS: {passed}")
    print(f"FAIL: {failed}")
    print("=" * 70)


if __name__ == "__main__":
    main()