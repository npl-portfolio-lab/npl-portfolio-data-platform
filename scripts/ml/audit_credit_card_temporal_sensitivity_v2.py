"""Fase 10.17: auditoría temporal de credit_card_balance V2.

Auditoría de solo lectura. No modifica datasets ni modelos.
"""

import json
from pathlib import Path

import duckdb

from npl_portfolio.core.paths import PROJECT_ROOT


SOURCE = (
    PROJECT_ROOT
    / "data/processed/home_credit/credit_card_balance.parquet"
)

CUTOFF_MONTH = -2

REQUIRED_COLUMNS = {
    "SK_ID_CURR",
    "SK_ID_PREV",
    "MONTHS_BALANCE",
    "SK_DPD",
    "AMT_BALANCE",
    "AMT_CREDIT_LIMIT_ACTUAL",
    "AMT_PAYMENT_TOTAL_CURRENT",
}

METRICS = {
    "record_count": "COUNT(*)",
    "max_dpd": "MAX(SK_DPD)",
    "avg_dpd": "AVG(SK_DPD)",
    "dpd_rate": (
        "COUNT(*) FILTER (WHERE SK_DPD > 0) "
        "* 1.0 / COUNT(*)"
    ),
    "avg_balance": "AVG(AMT_BALANCE)",
    "max_balance": "MAX(AMT_BALANCE)",
    "avg_credit_limit": "AVG(AMT_CREDIT_LIMIT_ACTUAL)",
    "avg_utilization": (
        "AVG(CASE WHEN AMT_CREDIT_LIMIT_ACTUAL > 0 "
        "THEN AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL "
        "ELSE NULL END)"
    ),
    "total_payment": "SUM(AMT_PAYMENT_TOTAL_CURRENT)",
}


def audit(source: Path | None = None) -> dict:
    source = Path(source) if source is not None else SOURCE

    if not source.is_file():
        raise FileNotFoundError(source)

    con = duckdb.connect()
    con.execute("SET threads = 1")

    try:
        available = {
            row[0]
            for row in con.execute(
                "DESCRIBE SELECT * FROM read_parquet(?)",
                [str(source)],
            ).fetchall()
        }

        missing = REQUIRED_COLUMNS - available

        if missing:
            raise ValueError(
                f"Columnas faltantes: {sorted(missing)}"
            )

        profile_result = con.execute(
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
            [str(source)],
        )

        profile_columns = [
            column[0] for column in profile_result.description
        ]

        temporal_profile = dict(
            zip(profile_columns, profile_result.fetchone())
        )

        aggregations = ",\n".join(
            f"{expression} AS {name}"
            for name, expression in METRICS.items()
        )

        comparisons = ",\n".join(
            (
                f"COUNT(*) FILTER ("
                f"WHERE h.SK_ID_CURR IS NOT NULL "
                f"AND o.{name} IS DISTINCT FROM h.{name}"
                f") AS changed_{name}"
            )
            for name in METRICS
        )

        query = f"""
            WITH original AS (
                SELECT SK_ID_CURR, {aggregations}
                FROM read_parquet(?)
                GROUP BY SK_ID_CURR
            ),
            historical AS (
                SELECT SK_ID_CURR, {aggregations}
                FROM read_parquet(?)
                WHERE MONTHS_BALANCE <= ?
                GROUP BY SK_ID_CURR
            )
            SELECT
                COUNT(*) AS total_clients,
                COUNT(h.SK_ID_CURR) AS comparable_clients,
                COUNT(*) FILTER (
                    WHERE h.SK_ID_CURR IS NULL
                ) AS clients_without_history,
                {comparisons}
            FROM original o
            LEFT JOIN historical h
                ON o.SK_ID_CURR = h.SK_ID_CURR
        """

        sensitivity_result = con.execute(
            query,
            [str(source), str(source), CUTOFF_MONTH],
        )

        sensitivity_columns = [
            column[0] for column in sensitivity_result.description
        ]

        sensitivity = dict(
            zip(
                sensitivity_columns,
                sensitivity_result.fetchone(),
            )
        )

        return {
            "phase": "10.17",
            "source": str(source),
            "cutoff_month": CUTOFF_MONTH,
            "temporal_profile": temporal_profile,
            "sensitivity": sensitivity,
            "risk_classification": (
                "Sensibilidad temporal identificada; "
                "Data Leakage no confirmado"
            ),
            "limitations": [
                "No demuestra disponibilidad punto-en-tiempo.",
                "No evalúa todas las características del constructor.",
                "No modifica ni reentrena modelos ML V2.",
                "Las diferencias de características se calculan "
                "solo para clientes con historial en ambos escenarios.",
                "Las comparaciones de promedios usan "
                "IS DISTINCT FROM sin tolerancia numérica.",
            ],
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
