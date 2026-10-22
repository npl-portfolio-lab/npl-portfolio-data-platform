import argparse
import getpass
from pathlib import Path

import duckdb
import psycopg
from psycopg import sql

BATCH_SIZE = 5000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["train", "test"])
    args = parser.parse_args()

    dataset = args.dataset
    path = Path(
        f"data/processed/features/client_features_{dataset}.parquet"
    )

    if not path.is_file():
        raise FileNotFoundError(path)

    password = getpass.getpass("Contraseña de npl_app: ")

    duck = duckdb.connect()
    duck.execute("SET threads = 2")

    try:
        columns = [
            row[0]
            for row in duck.execute(
                "DESCRIBE SELECT * FROM read_parquet(?)",
                [str(path)],
            ).fetchall()
        ]

        expected = duck.execute(
            "SELECT COUNT(*) FROM read_parquet(?)",
            [str(path)],
        ).fetchone()[0]

        table = sql.Identifier("npl", f"client_features_{dataset}")

        with psycopg.connect(
            dbname="npl_portfolio_db",
            user="npl_app",
            password=password,
            host="127.0.0.1",
            port=5432,
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("SELECT COUNT(*) FROM {}").format(table)
                )
                existing = cur.fetchone()[0]

                if existing != 0:
                    raise RuntimeError(
                        f"La tabla ya contiene {existing} registros. "
                        "No se realizará la carga."
                    )

                column_sql = sql.SQL(", ").join(
                    sql.Identifier(name) for name in columns
                )

                copy_sql = sql.SQL(
                    "COPY {} ({}) FROM STDIN"
                ).format(table, column_sql)

                reader = duck.execute(
                    "SELECT * FROM read_parquet(?)",
                    [str(path)],
                ).to_arrow_reader(batch_size=BATCH_SIZE)

                loaded = 0

                with cur.copy(copy_sql) as copy:
                    for batch in reader:
                        for row in batch.to_pylist():
                            copy.write_row(tuple(row[name] for name in columns))

                        loaded += batch.num_rows
                        print(
                            f"\r{dataset.upper()}: "
                            f"{loaded:,}/{expected:,}",
                            end="",
                            flush=True,
                        )

                print()

                cur.execute(
                    sql.SQL("SELECT COUNT(*) FROM {}").format(table)
                )
                actual = cur.fetchone()[0]

                if actual != expected:
                    raise RuntimeError(
                        f"Conteo incorrecto: {actual} != {expected}"
                    )

            print(f"CARGA VALIDADA: {actual:,} registros")

    finally:
        duck.close()


if __name__ == "__main__":
    main()
