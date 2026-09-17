from pathlib import Path

import pandas as pd
import pytest

from npl_portfolio.features.bureau_balance_features import (
    BureauBalanceFeatureBuilder,
)


def test_bureau_balance_feature_builder(
    tmp_path: Path,
) -> None:
    bureau_balance_path = (
        tmp_path / "bureau_balance.parquet"
    )
    bureau_path = tmp_path / "bureau.parquet"

    bureau = pd.DataFrame(
        {
            "SK_ID_BUREAU": [
                200001,
                200002,
                200003,
            ],
            "SK_ID_CURR": [
                100001,
                100001,
                100002,
            ],
        }
    )

    bureau_balance = pd.DataFrame(
        {
            "SK_ID_BUREAU": [
                200001,
                200001,
                200002,
                200002,
                200003,
                999999,
            ],
            "MONTHS_BALANCE": [
                -3,
                -2,
                -2,
                -1,
                -1,
                -5,
            ],
            "STATUS": [
                "0",
                "1",
                "3",
                "C",
                "X",
                "5",
            ],
        }
    )

    bureau.to_parquet(
        bureau_path,
        index=False,
    )

    bureau_balance.to_parquet(
        bureau_balance_path,
        index=False,
    )

    builder = BureauBalanceFeatureBuilder(
        parquet_path=bureau_balance_path,
        bureau_path=bureau_path,
    )

    result = builder.build()

    assert len(result) == 2
    assert result["SK_ID_CURR"].is_unique

    client_1 = result.loc[
        result["SK_ID_CURR"] == 100001
    ].iloc[0]

    assert client_1["BB_RECORD_COUNT"] == 4
    assert client_1["BB_BUREAU_CREDIT_COUNT"] == 2

    assert client_1["BB_OLDEST_MONTH"] == -3
    assert client_1["BB_RECENT_MONTH"] == -1

    assert client_1["BB_STATUS_0_COUNT"] == 1
    assert client_1["BB_STATUS_1_COUNT"] == 1
    assert client_1["BB_STATUS_2_COUNT"] == 0
    assert client_1["BB_STATUS_3_COUNT"] == 1
    assert client_1["BB_STATUS_4_COUNT"] == 0
    assert client_1["BB_STATUS_5_COUNT"] == 0
    assert client_1["BB_STATUS_C_COUNT"] == 1
    assert client_1["BB_STATUS_X_COUNT"] == 0

    assert client_1["BB_DPD_RECORD_COUNT"] == 2
    assert client_1[
        "BB_SEVERE_DPD_RECORD_COUNT"
    ] == 1

    assert client_1["BB_MAX_STATUS_LEVEL"] == 3

    assert client_1["BB_DPD_RATE"] == pytest.approx(
        2 / 4
    )

    assert client_1[
        "BB_SEVERE_DPD_RATE"
    ] == pytest.approx(
        1 / 4
    )

    assert client_1[
        "BB_CLOSED_STATUS_RATE"
    ] == pytest.approx(
        1 / 4
    )

    assert client_1[
        "BB_UNKNOWN_STATUS_RATE"
    ] == pytest.approx(
        0.0
    )

    client_2 = result.loc[
        result["SK_ID_CURR"] == 100002
    ].iloc[0]

    assert client_2["BB_RECORD_COUNT"] == 1
    assert client_2["BB_BUREAU_CREDIT_COUNT"] == 1
    assert client_2["BB_STATUS_X_COUNT"] == 1

    assert pd.isna(
        client_2["BB_MAX_STATUS_LEVEL"]
    )

    assert client_2["BB_DPD_RATE"] == pytest.approx(
        0.0
    )

    assert client_2[
        "BB_UNKNOWN_STATUS_RATE"
    ] == pytest.approx(
        1.0
    )

    # SK_ID_BUREAU 999999 no existe en bureau.
    # Por tanto no puede mapearse a SK_ID_CURR.
    assert result["BB_RECORD_COUNT"].sum() == 5
