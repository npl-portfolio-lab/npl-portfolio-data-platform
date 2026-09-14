from npl_portfolio.data_quality.rules import (
    DataQualityRule,
    DuplicateRowsRule,
    NullPercentageRule,
    RequiredColumnRule,
    UniqueColumnRule,
)


def build_home_credit_rules() -> dict[str, list[DataQualityRule]]:
    return {
        "application_train.csv": [
            RequiredColumnRule("SK_ID_CURR"),
            RequiredColumnRule("TARGET"),
            NullPercentageRule("SK_ID_CURR"),
            NullPercentageRule("TARGET"),
            UniqueColumnRule("SK_ID_CURR"),
            DuplicateRowsRule(),
        ],
        "application_test.csv": [
            RequiredColumnRule("SK_ID_CURR"),
            NullPercentageRule("SK_ID_CURR"),
            UniqueColumnRule("SK_ID_CURR"),
            DuplicateRowsRule(),
        ],
        "bureau.csv": [
            RequiredColumnRule("SK_ID_CURR"),
            RequiredColumnRule("SK_ID_BUREAU"),
            NullPercentageRule("SK_ID_CURR"),
            NullPercentageRule("SK_ID_BUREAU"),
            UniqueColumnRule("SK_ID_BUREAU"),
            DuplicateRowsRule(),
        ],
        "bureau_balance.csv": [
            RequiredColumnRule("SK_ID_BUREAU"),
            NullPercentageRule("SK_ID_BUREAU"),
            DuplicateRowsRule(),
        ],
        "previous_application.csv": [
            RequiredColumnRule("SK_ID_CURR"),
            RequiredColumnRule("SK_ID_PREV"),
            NullPercentageRule("SK_ID_CURR"),
            NullPercentageRule("SK_ID_PREV"),
            UniqueColumnRule("SK_ID_PREV"),
            DuplicateRowsRule(),
        ],
        "POS_CASH_balance.csv": [
            RequiredColumnRule("SK_ID_CURR"),
            RequiredColumnRule("SK_ID_PREV"),
            NullPercentageRule("SK_ID_CURR"),
            NullPercentageRule("SK_ID_PREV"),
            DuplicateRowsRule(),
        ],
        "credit_card_balance.csv": [
            RequiredColumnRule("SK_ID_CURR"),
            RequiredColumnRule("SK_ID_PREV"),
            NullPercentageRule("SK_ID_CURR"),
            NullPercentageRule("SK_ID_PREV"),
            DuplicateRowsRule(),
        ],
        "installments_payments.csv": [
            RequiredColumnRule("SK_ID_CURR"),
            RequiredColumnRule("SK_ID_PREV"),
            NullPercentageRule("SK_ID_CURR"),
            NullPercentageRule("SK_ID_PREV"),
            DuplicateRowsRule(),
        ],
    }
