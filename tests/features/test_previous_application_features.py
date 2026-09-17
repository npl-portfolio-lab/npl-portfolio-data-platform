from pathlib import Path

import pandas as pd
import pytest

from npl_portfolio.features.previous_application_features import (
    PreviousApplicationFeatureBuilder,
)


def test_previous_application_feature_builder(
    tmp_path: Path,
) -> None:
    parquet_path = tmp_path / "previous_application.parquet"

    data = pd.DataFrame(
        {
            "SK_ID_CURR": [
                100001,
                100001,
                100002,
                100003,
                100003,
            ],
            "NAME_CONTRACT_STATUS": [
                "Approved",
                "Refused",
                "Canceled",
                "Approved",
                "Unused offer",
            ],
            "AMT_APPLICATION": [
                1000.0,
                2000.0,
                500.0,
                None,
                None,
            ],
            "AMT_CREDIT": [
                900.0,
                1500.0,
                400.0,
                None,
                None,
            ],
            "AMT_ANNUITY": [
                100.0,
                200.0,
                50.0,
                80.0,
                90.0,
            ],
            "CNT_PAYMENT": [
                10.0,
                20.0,
                5.0,
                8.0,
                9.0,
            ],
            "DAYS_DECISION": [
                -100,
                -50,
                -20,
                -40,
                -10,
            ],
        }
    )

    data.to_parquet(
        parquet_path,
        index=False,
    )

    builder = PreviousApplicationFeatureBuilder(
        parquet_path=parquet_path,
    )

    result = builder.build()

    assert len(result) == 3
    assert result["SK_ID_CURR"].is_unique

    assert list(result.columns) == [
        "SK_ID_CURR",
        *PreviousApplicationFeatureBuilder.FEATURE_COLUMNS,
    ]

    client = result.loc[
        result["SK_ID_CURR"] == 100001
    ].iloc[0]

    assert client["PREV_APPLICATION_COUNT"] == 2
    assert client["PREV_APPROVED_COUNT"] == 1
    assert client["PREV_REFUSED_COUNT"] == 1
    assert client["PREV_CANCELED_COUNT"] == 0

    assert client[
        "PREV_AVG_APPLICATION_AMOUNT"
    ] == pytest.approx(1500.0)

    assert client[
        "PREV_TOTAL_APPLICATION_AMOUNT"
    ] == pytest.approx(3000.0)

    assert client[
        "PREV_TOTAL_CREDIT_AMOUNT"
    ] == pytest.approx(2400.0)

    assert client[
        "PREV_AVG_CREDIT_DIFFERENCE"
    ] == pytest.approx(300.0)

    assert client[
        "PREV_APPROVAL_RATE"
    ] == pytest.approx(0.5)

    assert client[
        "PREV_REFUSAL_RATE"
    ] == pytest.approx(0.5)

    null_client = result.loc[
        result["SK_ID_CURR"] == 100003
    ].iloc[0]

    assert pd.isna(
        null_client[
            "PREV_TOTAL_APPLICATION_AMOUNT"
        ]
    )

    assert pd.isna(
        null_client[
            "PREV_TOTAL_CREDIT_AMOUNT"
        ]
    )
