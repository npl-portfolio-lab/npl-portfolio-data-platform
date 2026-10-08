from sklearn.linear_model import LogisticRegression

from npl_portfolio.ml.baseline_trainer import (
    BaselineTrainer,
)


class BalancedTrainer(BaselineTrainer):
    """
    Logistic Regression con balanceo automático de clases.

    Mantiene la configuración del baseline original
    y modifica únicamente class_weight.
    """

    def __init__(
        self,
        random_state: int = 42,
        threshold: float = 0.50,
    ) -> None:
        super().__init__(
            random_state=random_state,
            threshold=threshold,
        )

        self.model = LogisticRegression(
            solver="liblinear",
            max_iter=1000,
            random_state=random_state,
            class_weight="balanced",
        )
