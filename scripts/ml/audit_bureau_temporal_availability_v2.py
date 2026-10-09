"""Fase 10.14: auditoría de disponibilidad temporal de Bureau.

Solo lectura. No modifica los datasets ni los modelos V2.
"""

import json
from pathlib import Path

import duckdb

from npl_portfolio.core.paths import PROJECT_ROOT


BUREAU = (
    PROJECT_ROOT / "data/processed/home_credit/bureau.parquet"
)

BUREAU_FEATURES = (
    PROJECT_ROOT / "data/interim/features/bureau_features.parquet"
)

SPLITS = {
    "TRAIN": PROJECT_ROOT / "data/processed/ml/v2/ml_train.parquet",
    "VALIDATION": PROJECT_ROOT / "data/processed/ml/v2/ml_validation.parquet",
    "TEST": PROJECT_ROOT / "data/processed/ml/v2/ml_test.parquet",
}


def audit():
    for path in [BUREAU, BUREAU_FEATURES, *SPLITS.values()]:
        if not path.is_file():
            raise FileNotFoundError(path)

    con = duckdb.connect()

    try:
        con.execute(
            """
            CREATE TEMP TABLE affected AS
            SELECT *
            FROM read_parquet(?)
            WHERE DAYS_CREDIT_UPDATE > 0
            """,
            [str(BUREAU)],
        )

        result = con.execute(
            """
            SELECT
                COUNT(*) AS records,
                COUNT(DISTINCT SK_ID_CURR) AS clients,
                COUNT(DISTINCT SK_ID_BUREAU) AS credits,
                MIN(DAYS_CREDIT_UPDATE) AS min_update,
                MAX(DAYS_CREDIT_UPDATE) AS max_update,
                COUNT(*) FILTER (
                    WHERE CREDIT_ACTIVE = 'Active'
                ) AS active,
                COUNT(*) FILTER (
                    WHERE CREDIT_ACTIVE = 'Closed'
                ) AS closed,
                COUNT(*) FILTER (
                    WHERE AMT_CREDIT_SUM_DEBT > 0
                ) AS with_debt,
                COUNT(*) FILTER (
                    WHERE CREDIT_DAY_OVERDUE > 0
                ) AS with_overdue_days,
                COUNT(*) FILTER (
                    WHERE AMT_CREDIT_SUM_OVERDUE > 0
                ) AS with_overdue_amount,
                COUNT(*) FILTER (
                    WHERE DAYS_CREDIT_ENDDATE > 0
                ) AS future_enddate
            FROM affected
            """
        ).fetchone()

        keys = [
            "records",
            "clients",
            "credits",
            "min_update",
            "max_update",
            "active",
            "closed",
            "with_debt",
            "with_overdue_days",
            "with_overdue_amount",
            "future_enddate",
        ]

        evidence = dict(zip(keys, result))

        evidence["clients_in_bureau_features"] = con.execute(
            """
            SELECT COUNT(DISTINCT a.SK_ID_CURR)
            FROM affected a
            INNER JOIN read_parquet(?) f
                ON a.SK_ID_CURR = f.SK_ID_CURR
            """,
            [str(BUREAU_FEATURES)],
        ).fetchone()[0]

        evidence["splits"] = {}

        for split_name, split_path in SPLITS.items():
            columns = {
                row[0]
                for row in con.execute(
                    "DESCRIBE SELECT * FROM read_parquet(?)",
                    [str(split_path)],
                ).fetchall()
            }

            if "SK_ID_CURR" not in columns:
                raise ValueError(
                    f"{split_name} no contiene SK_ID_CURR"
                )

            count = con.execute(
                """
                SELECT COUNT(DISTINCT a.SK_ID_CURR)
                FROM affected a
                INNER JOIN read_parquet(?) m
                    ON a.SK_ID_CURR = m.SK_ID_CURR
                """,
                [str(split_path)],
            ).fetchone()[0]

            evidence["splits"][split_name] = count

        evidence["phase"] = "10.14"
        evidence["risk_classification"] = (
            "Riesgo temporal potencial; "
            "Data Leakage no confirmado"
        )

        return evidence

    finally:
        con.close()


if __name__ == "__main__":
    result = audit()
    print("========== FASE 10.14 ==========")
    print(json.dumps(result, indent=2, ensure_ascii=False))
