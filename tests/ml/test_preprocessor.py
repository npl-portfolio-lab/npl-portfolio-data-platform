import numpy as np
import pandas as pd

from npl_portfolio.ml.preprocessor import MLPreprocessor



def test_ml_preprocessor() -> None:
    train = pd.DataFrame(
        {
            "SK_ID_CURR": [
                100001,
                100002,
                100003,
                100004,
            ],
            "TARGET": [
                0,
                1,
                0,
                0,
            ],
            "AMT_INCOME_TOTAL": [
                100000.0,
                200000.0,
                np.nan,
                150000.0,
            ],
            "AGE_YEARS": [
                30.0,
                40.0,
                50.0,
                np.nan,
            ],
            "NAME_CONTRACT_TYPE": [
                "Cash loans",
                "Cash loans",
                None,
                "Revolving loans",
            ],
        }
    )

    preprocessor_builder = MLPreprocessor()

    groups = preprocessor_builder.get_feature_groups(
        train
    )

    assert set(groups.numeric) == {
        "AMT_INCOME_TOTAL",
        "AGE_YEARS",
    }

    assert groups.categorical == [
        "NAME_CONTRACT_TYPE"
    ]

    assert "SK_ID_CURR" not in groups.numeric
    assert "TARGET" not in groups.numeric

    preprocessor = preprocessor_builder.build(
        train
    )

    features = train.drop(
        columns=[
            "SK_ID_CURR",
            "TARGET",
        ]
    )

    transformed = preprocessor.fit_transform(
        features
    )

    assert transformed.shape[0] == 4

    if hasattr(transformed, "data"):
        assert not np.isnan(
            transformed.data
        ).any()
    else:
        assert not np.isnan(
            transformed
        ).any()
