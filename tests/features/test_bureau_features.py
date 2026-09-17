from pathlib import Path

import pandas as pd
import pytest

from npl_portfolio.features.bureau_features import (
    BureauFeatureBuilder,
)


def test_bureau_feature_builder(
    tmp_path: Path,
) -> None:
    bureau_path = tmp_path / "bureau.parquet"

    bureau = pd.DataFrame(
        {
            "SK_ID_CURR": [
                100001,
                100001,
                100002,
            ],
            "CREDIT_ACTIVE": [
                "Active",
                "Closed",
                "Active",
            ],
            "CREDIT_DAY_OVERDUE": [
                10,
                0,
                0,
            ],
            "DAYS_CREDIT": [
                -100,
                -300,
                -50,
            ],
            "AMT_CREDIT_SUM": [
                1000.0,
                2000.0,
                500.0,
            ],
            "AMT_CREDIT_SUM_DEBT": [
                500.0,
                0.0,
                100.0,
            ],
            "AMT_CREDIT_SUM_OVERDUE": [
                50.0,
                0.0,
                0.0,
            ],
        }
    )

    bureau.to_parquet(
        bureau_path,
        index=False,
    )

    builder = BureauFeatureBuilder(
        parquet_path=bureau_path,
    )

    result = builder.build()

    assert len(result) == 2
    assert result["SK_ID_CURR"].is_unique

    assert list(result.columns) == [
        "SK_ID_CURR",
        *BureauFeatureBuilder.FEATURE_COLUMNS,
    ]

    client_1 = result.loc[
        result["SK_ID_CURR"] == 100001
    ].iloc[0]

    assert client_1[
        "BUREAU_CREDIT_COUNT"
    ] == 2

    assert client_1[
        "BUREAU_ACTIVE_COUNT"
    ] == 1

    assert client_1[
        "BUREAU_CLOSED_COUNT"
    ] == 1

    assert client_1[
        "BUREAU_DAYS_OVERDUE_COUNT"
    ] == 1

    assert client_1[
        "BUREAU_AMOUNT_OVERDUE_COUNT"
    ] == 1

    assert client_1[
        "BUREAU_AVG_DAYS_CREDIT"
    ] == pytest.approx(-200.0)

    assert client_1[
        "BUREAU_MIN_DAYS_CREDIT"
    ] == -300

    assert client_1[
        "BUREAU_TOTAL_CREDIT"
    ] == pytest.approx(3000.0)

    assert client_1[
        "BUREAU_TOTAL_DEBT"
    ] == pytest.approx(500.0)

    assert client_1[
        "BUREAU_TOTAL_OVERDUE"
    ] == pytest.approx(50.0)

    assert client_1[
        "BUREAU_MAX_OVERDUE_DAYS"
    ] == 10

    client_2 = result.loc[
        result["SK_ID_CURR"] == 100002
    ].iloc[0]

    assert client_2[
        "BUREAU_CREDIT_COUNT"
    ] == 1

    assert client_2[
        "BUREAU_ACTIVE_COUNT"
    ] == 1

    assert client_2[
        "BUREAU_DAYS_OVERDUE_COUNT"
    ] == 0
