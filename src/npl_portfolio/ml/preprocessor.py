from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass(frozen=True)
class FeatureGroups:
    numeric: list[str]
    categorical: list[str]


class MLPreprocessor:
    """
    Construye el pipeline de preprocessing para Machine Learning.

    Responsabilidades:
    - identificar features numéricas y categóricas;
    - excluir identificador y TARGET;
    - imputar valores faltantes;
    - codificar variables categóricas;
    - evitar data leakage.

    El pipeline debe ajustarse exclusivamente con TRAIN.
    """

    EXCLUDED_COLUMNS = {
        "SK_ID_CURR",
        "TARGET",
    }

    def get_feature_groups(
        self,
        dataframe: pd.DataFrame,
    ) -> FeatureGroups:
        feature_columns = [
            column
            for column in dataframe.columns
            if column not in self.EXCLUDED_COLUMNS
        ]

        features = dataframe[feature_columns]

        numeric_columns = features.select_dtypes(
            include="number",
        ).columns.tolist()

        categorical_columns = features.select_dtypes(
            include=["object", "string", "category"],
        ).columns.tolist()

        classified_columns = set(numeric_columns + categorical_columns)

        unclassified_columns = set(feature_columns) - classified_columns

        if unclassified_columns:
            raise ValueError(
                "Existen columnas con tipos no soportados: "
                f"{sorted(unclassified_columns)}"
            )

        return FeatureGroups(
            numeric=numeric_columns,
            categorical=categorical_columns,
        )

    def build(
        self,
        dataframe: pd.DataFrame,
    ) -> ColumnTransformer:
        groups = self.get_feature_groups(dataframe)

        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                    ),
                ),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent",
                    ),
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=True,
                    ),
                ),
            ]
        )

        return ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    numeric_pipeline,
                    groups.numeric,
                ),
                (
                    "categorical",
                    categorical_pipeline,
                    groups.categorical,
                ),
            ],
            remainder="drop",
            sparse_threshold=1.0,
        )
