from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from npl_portfolio.features.application_features import (
    ApplicationFeatureBuilder,
)


def test_application_feature_builder(
    tmp_path: Path,
) -> None:
    parquet_path = tmp_path / "application_train.parquet"

    application = pd.DataFrame(
        {
            "SK_ID_CURR": [
                100001,
                100002,
            ],
            "TARGET": [
                0,
                1,
            ],
            "DAYS_BIRTH": [
                -14610,
                -18262,
            ],
            "DAYS_EMPLOYED": [
                -3652,
                365243,
            ],
            "AMT_INCOME_TOTAL": [
                100000.0,
                0.0,
            ],
            "AMT_CREDIT": [
                500000.0,
                200000.0,
            ],
            "AMT_ANNUITY": [
                25000.0,
                0.0,
            ],
            "AMT_GOODS_PRICE": [
                450000.0,
                180000.0,
            ],
            "NAME_CONTRACT_TYPE": [
                "Cash loans",
                "Revolving loans",
            ],
        }
    )

    application.to_parquet(
        parquet_path,
        index=False,
    )

    builder = ApplicationFeatureBuilder(
        parquet_path=parquet_path,
    )

    result = builder.build()

    assert len(result) == 2
    assert result["SK_ID_CURR"].is_unique

    # TARGET no puede filtrarse hacia las features.
    assert "TARGET" not in result.columns

    for feature in ApplicationFeatureBuilder.DERIVED_FEATURES:
        assert feature in result.columns

    normal_client = result.loc[
        result["SK_ID_CURR"] == 100001
    ].iloc[0]

    anomaly_client = result.loc[
        result["SK_ID_CURR"] == 100002
    ].iloc[0]

    assert normal_client[
        "DAYS_EMPLOYED_ANOMALY"
    ] == 0

    assert anomaly_client[
        "DAYS_EMPLOYED_ANOMALY"
    ] == 1

    assert pd.isna(
        anomaly_client["DAYS_EMPLOYED"]
    )

    assert normal_client[
        "AGE_YEARS"
    ] == pytest.approx(
        14610 / 365.25
    )

    assert normal_client[
        "EMPLOYMENT_YEARS"
    ] == pytest.approx(
        3652 / 365.25
    )

    assert normal_client[
        "CREDIT_INCOME_RATIO"
    ] == pytest.approx(5.0)

    assert normal_client[
        "ANNUITY_INCOME_RATIO"
    ] == pytest.approx(0.25)

    assert normal_client[
        "CREDIT_ANNUITY_RATIO"
    ] == pytest.approx(20.0)

    assert normal_client[
        "GOODS_CREDIT_RATIO"
    ] == pytest.approx(0.9)

    # División por cero: nunca queremos infinito.
    assert pd.isna(
        anomaly_client[
            "CREDIT_INCOME_RATIO"
        ]
    )

    assert pd.isna(
        anomaly_client[
            "ANNUITY_INCOME_RATIO"
        ]
    )

    assert pd.isna(
        anomaly_client[
            "CREDIT_ANNUITY_RATIO"
        ]
    )

    numeric = result.select_dtypes(
        include=[np.number]
    )

    assert not np.isinf(
        numeric.to_numpy()
    ).any()
