from pathlib import Path

import pandas as pd

from npl_portfolio.features.pos_cash_features import (
    POSCashFeatureBuilder,
)


def test_pos_cash_feature_builder(tmp_path: Path) -> None:
    source_path = tmp_path / "pos_cash.parquet"

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
            "CNT_INSTALMENT": [
                12.0,
                12.0,
                24.0,
            ],
            "CNT_INSTALMENT_FUTURE": [
                10.0,
                9.0,
                20.0,
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

    builder = POSCashFeatureBuilder(
        parquet_path=source_path,
    )

    result = builder.build()

    assert len(result) == 2

    assert result["SK_ID_CURR"].is_unique

    client_1 = result.loc[
        result["SK_ID_CURR"] == 100001
    ].iloc[0]

    assert client_1["POS_RECORD_COUNT"] == 2
    assert client_1["POS_CONTRACT_COUNT"] == 1
    assert client_1["POS_ACTIVE_RECORD_COUNT"] == 1
    assert client_1["POS_COMPLETED_RECORD_COUNT"] == 1

    assert client_1["POS_DPD_RECORD_COUNT"] == 1
    assert client_1["POS_DPD_DEF_RECORD_COUNT"] == 1

    assert client_1["POS_MAX_DPD"] == 10
    assert client_1["POS_MAX_DPD_DEF"] == 2

    assert client_1["POS_DPD_RATE"] == 0.5
    assert client_1["POS_DPD_DEF_RATE"] == 0.5

    client_2 = result.loc[
        result["SK_ID_CURR"] == 100002
    ].iloc[0]

    assert client_2["POS_RECORD_COUNT"] == 1
    assert client_2["POS_CONTRACT_COUNT"] == 1

    assert client_2["POS_DPD_RATE"] == 1.0
    assert client_2["POS_DPD_DEF_RATE"] == 0.0
