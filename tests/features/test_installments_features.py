from pathlib import Path

import pandas as pd
import pytest

from npl_portfolio.features.installments_features import (
    InstallmentsFeatureBuilder,
)


def test_installments_feature_builder(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "installments.parquet"

    source = pd.DataFrame(
        {
            "SK_ID_CURR": [
                100001,
                100001,
                100001,
                100002,
            ],
            "SK_ID_PREV": [
                200001,
                200001,
                200002,
                200003,
            ],
            "NUM_INSTALMENT_VERSION": [
                1.0,
                1.0,
                2.0,
                1.0,
            ],
            "NUM_INSTALMENT_NUMBER": [
                1,
                2,
                1,
                1,
            ],
            "DAYS_INSTALMENT": [
                -100,
                -50,
                -20,
                -30,
            ],
            "DAYS_ENTRY_PAYMENT": [
                -105.0,
                -40.0,
                None,
                -30.0,
            ],
            "AMT_INSTALMENT": [
                100.0,
                200.0,
                300.0,
                400.0,
            ],
            "AMT_PAYMENT": [
                100.0,
                150.0,
                None,
                500.0,
            ],
        }
    )

    source.to_parquet(
        source_path,
        index=False,
    )

    builder = InstallmentsFeatureBuilder(
        parquet_path=source_path,
    )

    result = builder.build()

    assert len(result) == 2
    assert result["SK_ID_CURR"].is_unique

    client_1 = result.loc[
        result["SK_ID_CURR"] == 100001
    ].iloc[0]

    assert client_1["INST_RECORD_COUNT"] == 3
    assert client_1["INST_CONTRACT_COUNT"] == 2

    assert client_1[
        "INST_TOTAL_INSTALMENT_AMOUNT"
    ] == pytest.approx(600.0)

    # SUM ignora el AMT_PAYMENT nulo:
    # 100 + 150 = 250
    assert client_1[
        "INST_TOTAL_PAYMENT_AMOUNT"
    ] == pytest.approx(250.0)

    # Delays válidos:
    # -105 - (-100) = -5
    # -40 - (-50) = 10
    # tercer registro = NULL
    assert client_1[
        "INST_AVG_PAYMENT_DELAY"
    ] == pytest.approx(2.5)

    assert client_1[
        "INST_MAX_PAYMENT_DELAY"
    ] == pytest.approx(10.0)

    assert client_1[
        "INST_MIN_PAYMENT_DELAY"
    ] == pytest.approx(-5.0)

    assert client_1[
        "INST_LATE_PAYMENT_COUNT"
    ] == 1

    assert client_1[
        "INST_ON_TIME_OR_EARLY_COUNT"
    ] == 1

    assert client_1[
        "INST_MISSING_PAYMENT_DATE_COUNT"
    ] == 1

    # Diferencias válidas:
    # 100 - 100 = 0
    # 200 - 150 = 50
    # tercer registro = NULL
    assert client_1[
        "INST_AVG_PAYMENT_DIFFERENCE"
    ] == pytest.approx(25.0)

    assert client_1[
        "INST_MAX_PAYMENT_DIFFERENCE"
    ] == pytest.approx(50.0)

    assert client_1[
        "INST_UNDERPAYMENT_COUNT"
    ] == 1

    assert client_1[
        "INST_OVERPAYMENT_COUNT"
    ] == 0

    # Las tasas mantienen exactamente la lógica del EDA:
    # denominador = todos los registros del cliente.
    assert client_1[
        "INST_LATE_PAYMENT_RATE"
    ] == pytest.approx(1 / 3)

    assert client_1[
        "INST_UNDERPAYMENT_RATE"
    ] == pytest.approx(1 / 3)

    assert client_1[
        "INST_OVERPAYMENT_RATE"
    ] == pytest.approx(0.0)

    assert client_1[
        "INST_PAYMENT_RATIO"
    ] == pytest.approx(250 / 600)

    client_2 = result.loc[
        result["SK_ID_CURR"] == 100002
    ].iloc[0]

    # Pago exactamente en fecha.
    assert client_2[
        "INST_LATE_PAYMENT_COUNT"
    ] == 0

    assert client_2[
        "INST_ON_TIME_OR_EARLY_COUNT"
    ] == 1

    # 400 esperado - 500 pagado = -100:
    # es un sobrepago.
    assert client_2[
        "INST_OVERPAYMENT_COUNT"
    ] == 1

    assert client_2[
        "INST_PAYMENT_RATIO"
    ] == pytest.approx(1.25)
