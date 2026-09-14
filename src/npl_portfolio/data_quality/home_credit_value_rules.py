from npl_portfolio.data_quality.value_rules import (
    AllowedValuesRule,
    NonNegativeValueRule,
    PositiveValueRule,
    ValueQualityRule,
)


def build_home_credit_value_rules() -> dict[str, list[ValueQualityRule]]:

    return {
        "application_train.csv": [
            AllowedValuesRule(
                column="TARGET",
                allowed_values={0, 1},
                allow_null=False,
            ),
            PositiveValueRule(
                column="SK_ID_CURR",
                allow_null=False,
            ),
            NonNegativeValueRule(
                column="AMT_INCOME_TOTAL",
                allow_null=False,
            ),
            NonNegativeValueRule(
                column="AMT_CREDIT",
                allow_null=False,
            ),
            NonNegativeValueRule(
                column="AMT_ANNUITY",
                allow_null=True,
            ),
            NonNegativeValueRule(
                column="AMT_GOODS_PRICE",
                allow_null=True,
            ),
        ],
        "application_test.csv": [
            PositiveValueRule(
                column="SK_ID_CURR",
                allow_null=False,
            ),
            NonNegativeValueRule(
                column="AMT_INCOME_TOTAL",
                allow_null=False,
            ),
            NonNegativeValueRule(
                column="AMT_CREDIT",
                allow_null=False,
            ),
            NonNegativeValueRule(
                column="AMT_ANNUITY",
                allow_null=True,
            ),
            NonNegativeValueRule(
                column="AMT_GOODS_PRICE",
                allow_null=True,
            ),
        ],
        "bureau.csv": [
            PositiveValueRule(
                column="SK_ID_CURR",
                allow_null=False,
            ),
            PositiveValueRule(
                column="SK_ID_BUREAU",
                allow_null=False,
            ),
        ],
        "bureau_balance.csv": [
            PositiveValueRule(
                column="SK_ID_BUREAU",
                allow_null=False,
            ),
        ],
        "previous_application.csv": [
            PositiveValueRule(
                column="SK_ID_CURR",
                allow_null=False,
            ),
            PositiveValueRule(
                column="SK_ID_PREV",
                allow_null=False,
            ),
        ],
        "POS_CASH_balance.csv": [
            PositiveValueRule(
                column="SK_ID_CURR",
                allow_null=False,
            ),
            PositiveValueRule(
                column="SK_ID_PREV",
                allow_null=False,
            ),
        ],
        "credit_card_balance.csv": [
            PositiveValueRule(
                column="SK_ID_CURR",
                allow_null=False,
            ),
            PositiveValueRule(
                column="SK_ID_PREV",
                allow_null=False,
            ),
        ],
        "installments_payments.csv": [
            PositiveValueRule(
                column="SK_ID_CURR",
                allow_null=False,
            ),
            PositiveValueRule(
                column="SK_ID_PREV",
                allow_null=False,
            ),
        ],
    }
