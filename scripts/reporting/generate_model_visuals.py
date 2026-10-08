from pathlib import Path
from npl_portfolio.core.paths import PROJECT_ROOT, ML_DATA_DIR

import joblib
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)

from npl_portfolio.ml.dataset_loader import SparseDatasetLoader

ROOT = PROJECT_ROOT

ML_DIR = ML_DATA_DIR

MODEL_PATH = ROOT / "data" / "artifacts" / "ml" / "baseline_logistic_regression.joblib"

OUT_DIR = ROOT / "artifacts" / "ml" / "linkedin"

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def get_estimator(trainer):
    """
    Encuentra dentro del BaselineTrainer el modelo sklearn
    que tiene predict_proba().
    """

    if hasattr(trainer, "predict_proba"):
        return trainer

    for attr_name in dir(trainer):
        if attr_name.startswith("_"):
            continue

        try:
            obj = getattr(trainer, attr_name)
        except Exception:
            continue

        if hasattr(obj, "predict_proba"):
            print(f"[OK] Estimador encontrado " f"en trainer.{attr_name}")
            return obj

    raise AttributeError(
        "No encontré un estimador con " "predict_proba() dentro del BaselineTrainer."
    )


def main():

    print("=" * 80)
    print("GENERANDO VISUALES DEL BASELINE")
    print("=" * 80)

    # =====================================================
    # CARGAR VALIDATION CON EL LOADER REAL DEL PROYECTO
    # =====================================================

    loader = SparseDatasetLoader(ml_dir=ML_DIR)

    print("\n[LOAD] VALIDATION")

    validation = loader.load("validation")

    X_val = validation.X
    y_val = np.asarray(validation.y).ravel()

    print(
        f"[OK] VALIDATION: "
        f"{X_val.shape[0]:,} filas x "
        f"{X_val.shape[1]:,} features"
    )

    print(f"[OK] TARGET positivos: " f"{y_val.sum():,}")

    # =====================================================
    # CARGAR BASELINE YA ENTRENADO
    # =====================================================

    print("\n[LOAD] MODELO")

    trainer = joblib.load(MODEL_PATH)

    print(f"[OK] Modelo cargado: " f"{MODEL_PATH}")

    estimator = get_estimator(trainer)

    # =====================================================
    # PROBABILIDADES
    # =====================================================

    print("\n[PREDICT] Calculando probabilidades...")

    y_proba = np.asarray(estimator.predict_proba(X_val))

    # BaselineTrainer puede devolver directamente
    # la probabilidad de la clase positiva.
    if y_proba.ndim == 2:
        y_proba = y_proba[:, 1]
    else:
        y_proba = y_proba.ravel()

    print(f"[OK] Probabilidades generadas: " f"{len(y_proba):,}")

    # =====================================================
    # 05 - ROC CURVE
    # =====================================================

    fpr, tpr, _ = roc_curve(
        y_val,
        y_proba,
    )

    roc_auc = auc(
        fpr,
        tpr,
    )

    plt.figure(figsize=(9, 6))

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"ROC-AUC = {roc_auc:.4f}",
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Clasificador aleatorio",
    )

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title("Baseline Logistic Regression - ROC Curve")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUT_DIR / "05_roc_curve.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("[OK] 05_roc_curve.png")

    # =====================================================
    # 06 - PRECISION RECALL
    # =====================================================

    precision_curve, recall_curve, _ = precision_recall_curve(
        y_val,
        y_proba,
    )

    pr_auc = average_precision_score(
        y_val,
        y_proba,
    )

    baseline_rate = y_val.mean()

    plt.figure(figsize=(9, 6))

    plt.plot(
        recall_curve,
        precision_curve,
        linewidth=2,
        label=f"PR-AUC = {pr_auc:.4f}",
    )

    plt.axhline(
        baseline_rate,
        linestyle="--",
        label=(f"Baseline clase positiva " f"= {baseline_rate:.2%}"),
    )

    plt.xlabel("Recall")

    plt.ylabel("Precision")

    plt.title("Baseline Logistic Regression - Precision-Recall Curve")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUT_DIR / "06_precision_recall_curve.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("[OK] 06_precision_recall_curve.png")

    # =====================================================
    # 07 - DISTRIBUCION DE PROBABILIDADES
    # =====================================================

    plt.figure(figsize=(10, 6))

    plt.hist(
        y_proba[y_val == 0],
        bins=60,
        alpha=0.60,
        label="TARGET = 0",
    )

    plt.hist(
        y_proba[y_val == 1],
        bins=60,
        alpha=0.60,
        label="TARGET = 1",
    )

    plt.axvline(
        0.50,
        linestyle="--",
        linewidth=2,
        label="Threshold = 0.50",
    )

    plt.xlabel("Probabilidad predicha de TARGET = 1")

    plt.ylabel("Número de observaciones")

    plt.title("Baseline Logistic Regression - Predicted Probabilities")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUT_DIR / "07_probability_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("[OK] 07_probability_distribution.png")

    # =====================================================
    # 08 - THRESHOLD VS METRICS
    # =====================================================

    thresholds = np.arange(
        0.01,
        0.51,
        0.01,
    )

    precisions = []
    recalls = []
    f1_scores = []

    for threshold in thresholds:

        y_pred = (y_proba >= threshold).astype(int)

        precisions.append(
            precision_score(
                y_val,
                y_pred,
                zero_division=0,
            )
        )

        recalls.append(
            recall_score(
                y_val,
                y_pred,
                zero_division=0,
            )
        )

        f1_scores.append(
            f1_score(
                y_val,
                y_pred,
                zero_division=0,
            )
        )

    plt.figure(figsize=(10, 6))

    plt.plot(
        thresholds,
        precisions,
        label="Precision",
        linewidth=2,
    )

    plt.plot(
        thresholds,
        recalls,
        label="Recall",
        linewidth=2,
    )

    plt.plot(
        thresholds,
        f1_scores,
        label="F1-score",
        linewidth=2,
    )

    plt.axvline(
        0.50,
        linestyle="--",
        label="Threshold actual = 0.50",
    )

    plt.xlabel("Threshold")

    plt.ylabel("Métrica")

    plt.title("Baseline Logistic Regression - Threshold Analysis")

    plt.legend()

    plt.grid(alpha=0.25)

    plt.tight_layout()

    plt.savefig(
        OUT_DIR / "08_threshold_metrics.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("[OK] 08_threshold_metrics.png")

    # =====================================================
    # FINAL
    # =====================================================

    print()
    print("=" * 80)
    print("VISUALES GENERADOS")
    print("=" * 80)

    print(f"ROC-AUC: {roc_auc:.6f}")

    print(f"PR-AUC:  {pr_auc:.6f}")

    print()

    for path in sorted(OUT_DIR.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
