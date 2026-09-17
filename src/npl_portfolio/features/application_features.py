from pathlib import Path

import numpy as np
import pandas as pd


class ApplicationFeatureBuilder:
    """
    Construye features determinísticas de application_train
    y application_test.

    Reglas:
    - Una fila por SK_ID_CURR.
    - TARGET nunca forma parte de las features.
    - DAYS_EMPLOYED=365243 se trata como valor sentinel.
    - No se realiza imputación aprendida.
    - No se realiza encoding aprendido.
    - No se utiliza información de TARGET.
    """

    DAYS_EMPLOYED_SENTINEL = 365243

    REQUIRED_COLUMNS = [
        "SK_ID_CURR",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
    ]

    DERIVED_FEATURES = [
        "DAYS_EMPLOYED_ANOMALY",
        "AGE_YEARS",
        "EMPLOYMENT_YEARS",
        "CREDIT_INCOME_RATIO",
        "ANNUITY_INCOME_RATIO",
        "CREDIT_ANNUITY_RATIO",
        "GOODS_CREDIT_RATIO",
    ]

    def __init__(
        self,
        parquet_path: Path,
    ) -> None:
        self.parquet_path = parquet_path

        if not self.parquet_path.exists():
            raise FileNotFoundError(
                f"No existe el archivo: {self.parquet_path}"
            )

    @staticmethod
    def _safe_divide(
        numerator: pd.Series,
        denominator: pd.Series,
    ) -> pd.Series:
        """
        División segura.

        Denominadores cero se convierten en NaN y cualquier
        infinito resultante se reemplaza por NaN.
        """
        safe_denominator = denominator.replace(
            0,
            np.nan,
        )

        result = numerator / safe_denominator

        return result.replace(
            [np.inf, -np.inf],
            np.nan,
        )

    def build(self) -> pd.DataFrame:
        application = pd.read_parquet(
            self.parquet_path,
        )

        missing_columns = [
            column
            for column in self.REQUIRED_COLUMNS
            if column not in application.columns
        ]

        if missing_columns:
            raise ValueError(
                "Faltan columnas requeridas: "
                f"{missing_columns}"
            )

        if not application["SK_ID_CURR"].is_unique:
            raise ValueError(
                "SK_ID_CURR contiene duplicados en application."
            )

        # TARGET nunca debe formar parte del dataset
        # generado por Feature Engineering.
        if "TARGET" in application.columns:
            application = application.drop(
                columns=["TARGET"]
            )

        # --------------------------------------------------
        # Sentinel DAYS_EMPLOYED
        # --------------------------------------------------

        application["DAYS_EMPLOYED_ANOMALY"] = (
            application["DAYS_EMPLOYED"]
            .eq(self.DAYS_EMPLOYED_SENTINEL)
            .astype("int8")
        )

        application.loc[
            application["DAYS_EMPLOYED"].eq(
                self.DAYS_EMPLOYED_SENTINEL
            ),
            "DAYS_EMPLOYED",
        ] = np.nan

        # --------------------------------------------------
        # Variables temporales
        # --------------------------------------------------

        application["AGE_YEARS"] = (
            -application["DAYS_BIRTH"] / 365.25
        )

        application["EMPLOYMENT_YEARS"] = (
            -application["DAYS_EMPLOYED"] / 365.25
        )

        # --------------------------------------------------
        # Ratios financieros
        # --------------------------------------------------

        application["CREDIT_INCOME_RATIO"] = (
            self._safe_divide(
                application["AMT_CREDIT"],
                application["AMT_INCOME_TOTAL"],
            )
        )

        application["ANNUITY_INCOME_RATIO"] = (
            self._safe_divide(
                application["AMT_ANNUITY"],
                application["AMT_INCOME_TOTAL"],
            )
        )

        application["CREDIT_ANNUITY_RATIO"] = (
            self._safe_divide(
                application["AMT_CREDIT"],
                application["AMT_ANNUITY"],
            )
        )

        application["GOODS_CREDIT_RATIO"] = (
            self._safe_divide(
                application["AMT_GOODS_PRICE"],
                application["AMT_CREDIT"],
            )
        )

        return application
