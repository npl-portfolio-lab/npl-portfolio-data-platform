from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def main() -> None:

    project_root = Path(__file__).resolve().parents[1]

    csv_directory = project_root / "data" / "raw" / "home_credit"

    parquet_directory = project_root / "data" / "processed" / "home_credit"

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

    print()
    print("VERIFICACIÓN CSV VS PARQUET")
    print("=" * 90)

    all_valid = True

    for csv_name in files:

        csv_path = csv_directory / csv_name

        parquet_name = Path(csv_name).with_suffix(".parquet").name

        parquet_path = parquet_directory / parquet_name

        csv_columns = pd.read_csv(
            csv_path,
            nrows=0,
        ).columns.tolist()

        parquet_file = pq.ParquetFile(parquet_path)

        parquet_rows = parquet_file.metadata.num_rows

        parquet_columns = parquet_file.schema_arrow.names

        # Contar filas del CSV sin cargarlo completo
        csv_rows = 0

        for chunk in pd.read_csv(
            csv_path,
            usecols=[csv_columns[0]],
            chunksize=500_000,
        ):
            csv_rows += len(chunk)

        rows_match = csv_rows == parquet_rows

        columns_match = csv_columns == parquet_columns

        valid = rows_match and columns_match

        if not valid:
            all_valid = False

        status = "PASS" if valid else "FAIL"

        print()
        print(f"[{status}] {csv_name}")

        print(f"  Filas CSV:     " f"{csv_rows:,}")

        print(f"  Filas Parquet: " f"{parquet_rows:,}")

        print(f"  Columnas CSV:     " f"{len(csv_columns)}")

        print(f"  Columnas Parquet: " f"{len(parquet_columns)}")

        print(f"  Filas iguales: " f"{rows_match}")

        print(f"  Columnas iguales: " f"{columns_match}")

    print()
    print("=" * 90)

    if all_valid:
        print("RESULTADO FINAL: " "TODAS LAS TRANSFORMACIONES SON VÁLIDAS")
    else:
        print("RESULTADO FINAL: " "SE ENCONTRARON DIFERENCIAS")


if __name__ == "__main__":
    main()
