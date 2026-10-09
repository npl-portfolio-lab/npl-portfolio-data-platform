"""Fase 10.16: auditoría temporal y sensibilidad POS_CASH V2.

Solo lectura. No modifica datasets, características ni modelos.
"""

import json

import duckdb

from npl_portfolio.core.paths import PROJECT_ROOT


SOURCE = (
    PROJECT_ROOT
    / "data/processed/home_credit/POS_CASH_balance.parquet"
)

CUTOFF_MONTH = -2


def audit():
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)

    con = duckdb.connect()

    try:
        required = {
            "SK_ID_CURR",
            "SK_ID_PREV",
            "MONTHS_BALANCE",
            "SK_DPD",
        }

        available = {
            row[0]
            for row in con.execute(
                "DESCRIBE SELECT * FROM read_parquet(?)",
                [str(SOURCE)],
            ).fetchall()
        }

        missing = required - available

        if missing:
            raise ValueError(
                f"Columnas faltantes: {sorted(missing)}"
            )

        profile = con.execute(
            """
            SELECT
                COUNT(*) AS records,
                COUNT(DISTINCT SK_ID_CURR) AS clients,
                COUNT(DISTINCT SK_ID_PREV) AS contracts,
                COUNT(*) FILTER (
                    WHERE MONTHS_BALANCE IS NULL
                ) AS null_months,
                COUNT(*) FILTER (
                    WHERE MONTHS_BALANCE >= 0
                ) AS nonhistorical_months,
                COUNT(*) FILTER (
                    WHERE MONTHS_BALANCE = -1
                ) AS recent_records,
                COUNT(DISTINCT SK_ID_CURR) FILTER (
                    WHERE MONTHS_BALANCE = -1
                ) AS recent_clients,
                MIN(MONTHS_BALANCE) AS oldest_month,
                MAX(MONTHS_BALANCE) AS newest_month
            FROM read_parquet(?)
            """,
            [str(SOURCE)],
        ).fetchone()

        profile_keys = [
            "records",
            "clients",
            "contracts",
            "null_months",
            "nonhistorical_months",
            "recent_records",
            "recent_clients",
            "oldest_month",
            "newest_month",
        ]

        temporal_profile = dict(zip(profile_keys, profile))

        sensitivity = con.execute(
            """
            WITH original AS (
                SELECT
                    SK_ID_CURR,
                    COUNT(*) AS record_count,
                    MAX(SK_DPD) AS max_dpd,
                    AVG(SK_DPD) AS avg_dpd,
                    COUNT(*) FILTER (
                        WHERE SK_DPD > 0
                    ) * 1.0 / COUNT(*) AS dpd_rate
                FROM read_parquet(?)
                GROUP BY SK_ID_CURR
            ),
            historical AS (
                SELECT
                    SK_ID_CURR,
                    COUNT(*) AS record_count,
                    MAX(SK_DPD) AS max_dpd,
                    AVG(SK_DPD) AS avg_dpd,
                    COUNT(*) FILTER (
                        WHERE SK_DPD > 0
                    ) * 1.0 / COUNT(*) AS dpd_rate
                FROM read_parquet(?)
                WHERE MONTHS_BALANCE <= ?
                GROUP BY SK_ID_CURR
            )
            SELECT
                COUNT(*) AS total_clients,
                COUNT(*) FILTER (
                    WHERE h.SK_ID_CURR IS NULL
                ) AS clients_without_history,
                COUNT(*) FILTER (
                    WHERE o.record_count
                    IS DISTINCT FROM h.record_count
                ) AS changed_record_count,
                COUNT(*) FILTER (
                    WHERE o.max_dpd
                    IS DISTINCT FROM h.max_dpd
                ) AS changed_max_dpd,
                COUNT(*) FILTER (
                    WHERE o.avg_dpd
                    IS DISTINCT FROM h.avg_dpd
                ) AS changed_avg_dpd,
                COUNT(*) FILTER (
                    WHERE o.dpd_rate
                    IS DISTINCT FROM h.dpd_rate
                ) AS changed_dpd_rate
            FROM original o
            LEFT JOIN historical h
                ON o.SK_ID_CURR = h.SK_ID_CURR
            """,
            [str(SOURCE), str(SOURCE), CUTOFF_MONTH],
        ).fetchone()

        sensitivity_keys = [
            "total_clients",
            "clients_without_history",
            "changed_record_count",
            "changed_max_dpd",
            "changed_avg_dpd",
            "changed_dpd_rate",
        ]

        return {
            "phase": "10.16",
            "source": str(SOURCE),
            "cutoff_month": CUTOFF_MONTH,
            "temporal_profile": temporal_profile,
            "sensitivity": dict(
                zip(sensitivity_keys, sensitivity)
            ),
            "risk_classification": (
                "Sensibilidad temporal identificada; "
                "Data Leakage no confirmado"
            ),
        }

    finally:
        con.close()


if __name__ == "__main__":
    print(
        json.dumps(
            audit(),
            indent=2,
            ensure_ascii=False,
        )
    )
