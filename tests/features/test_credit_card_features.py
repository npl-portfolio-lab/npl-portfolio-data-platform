from pathlib import Path

import pandas as pd
import pytest

from npl_portfolio.features.credit_card_features import (
    CreditCardFeatureBuilder,
)


def test_credit_card_feature_builder(tmp_path: Path) -> None:
    source_path = tmp_path / "credit_card.parquet"

    source = pd.DataFrame(
        {
            "SK_ID_CURR": [
                100001,
                100001,
                100002,
            ],
            "SK_ID_PREV": [
                200001,
                200001,
                200002,
            ],
            "NAME_CONTRACT_STATUS": [
                "Active",
                "Completed",
                "Active",
            ],
            "AMT_BALANCE": [
                500.0,
                1000.0,
                300.0,
            ],
            "AMT_CREDIT_LIMIT_ACTUAL": [
                1000.0,
                2000.0,
                0.0,
            ],
            "AMT_DRAWINGS_CURRENT": [
                100.0,
                200.0,
                50.0,
            ],
            "AMT_PAYMENT_CURRENT": [
                80.0,
                150.0,
                40.0,
            ],
            "AMT_PAYMENT_TOTAL_CURRENT": [
                90.0,
                160.0,
                45.0,
            ],
            "AMT_TOTAL_RECEIVABLE": [
                450.0,
                900.0,
                250.0,
            ],
            "CNT_DRAWINGS_CURRENT": [
                1.0,
                2.0,
                1.0,
            ],
            "CNT_INSTALMENT_MATURE_CUM": [
                2.0,
                3.0,
                1.0,
            ],
            "SK_DPD": [
                0,
                10,
                5,
            ],
            "SK_DPD_DEF": [
                0,
                2,
                0,
            ],
            "MONTHS_BALANCE": [
                -2,
                -1,
                -1,
            ],
        }
    )

    source.to_parquet(
        source_path,
        index=False,
    )

    builder = CreditCardFeatureBuilder(
        parquet_path=source_path,
    )

    result = builder.build()

    assert len(result) == 2
    assert result["SK_ID_CURR"].is_unique

    client_1 = result.loc[
        result["SK_ID_CURR"] == 100001
    ].iloc[0]

    assert client_1["CC_RECORD_COUNT"] == 2
    assert client_1["CC_CONTRACT_COUNT"] == 1

    assert client_1["CC_ACTIVE_RECORD_COUNT"] == 1
    assert client_1["CC_COMPLETED_RECORD_COUNT"] == 1

    assert client_1["CC_AVG_BALANCE"] == pytest.approx(750.0)
    assert client_1["CC_MAX_BALANCE"] == pytest.approx(1000.0)

    assert client_1["CC_TOTAL_DRAWINGS"] == pytest.approx(300.0)
    assert client_1["CC_TOTAL_PAYMENT"] == pytest.approx(250.0)

    assert client_1["CC_DPD_RECORD_COUNT"] == 1
    assert client_1["CC_DPD_DEF_RECORD_COUNT"] == 1

    assert client_1["CC_DPD_RATE"] == pytest.approx(0.5)
    assert client_1["CC_DPD_DEF_RATE"] == pytest.approx(0.5)

    assert client_1["CC_AVG_UTILIZATION"] == pytest.approx(0.5)
    assert client_1["CC_MAX_UTILIZATION"] == pytest.approx(0.5)

    client_2 = result.loc[
        result["SK_ID_CURR"] == 100002
    ].iloc[0]

    assert client_2["CC_RECORD_COUNT"] == 1
    assert client_2["CC_DPD_RATE"] == pytest.approx(1.0)
    assert client_2["CC_DPD_DEF_RATE"] == pytest.approx(0.0)

    assert pd.isna(
        client_2["CC_AVG_UTILIZATION"]
    )

    assert pd.isna(
        client_2["CC_MAX_UTILIZATION"]
    )
