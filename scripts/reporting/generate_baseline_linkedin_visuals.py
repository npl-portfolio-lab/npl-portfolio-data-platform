from pathlib import Path
from npl_portfolio.core.paths import VISUAL_ARTIFACTS_DIR
import matplotlib.pyplot as plt
import numpy as np

OUT = VISUAL_ARTIFACTS_DIR / "ml" / "linkedin"
OUT.mkdir(parents=True, exist_ok=True)

# =========================================================
# DATOS REALES DE LA FASE 9.4
# =========================================================

train_total = 246_008
train_positive = 19_860

val_total = 61_503
val_positive = 4_965

roc_auc = 0.682146
pr_auc = 0.157146
precision = 0.032258
recall = 0.000201
f1 = 0.000400

tn = 56_508
fp = 30
fn = 4_964
tp = 1


# =========================================================
# 1. DISTRIBUCIÓN TARGET
# =========================================================

train_negative = train_total - train_positive
val_negative = val_total - val_positive

labels = ["TRAIN", "VALIDATION"]
target_0 = [train_negative, val_negative]
target_1 = [train_positive, val_positive]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

bars0 = ax.bar(x - width/2, target_0, width, label="TARGET = 0")
bars1 = ax.bar(x + width/2, target_1, width, label="TARGET = 1")

ax.set_title("Distribución de TARGET - Fase 9.4")
ax.set_ylabel("Número de registros")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

for bars in [bars0, bars1]:
    for bar in bars:
        value = int(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width()/2,
            value,
            f"{value:,}",
            ha="center",
            va="bottom",
            fontsize=9
        )

plt.tight_layout()
plt.savefig(OUT / "01_target_distribution.png", dpi=300, bbox_inches="tight")
plt.close()


# =========================================================
# 2. MATRIZ DE CONFUSIÓN
# =========================================================

cm = np.array([
    [tn, fp],
    [fn, tp]
])

fig, ax = plt.subplots(figsize=(7, 6))

im = ax.imshow(cm)

ax.set_title("Baseline Logistic Regression - Matriz de Confusión")
ax.set_xlabel("Predicción")
ax.set_ylabel("Real")

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels(["TARGET 0", "TARGET 1"])
ax.set_yticklabels(["TARGET 0", "TARGET 1"])

for i in range(2):
    for j in range(2):
        ax.text(
            j,
            i,
            f"{cm[i, j]:,}",
            ha="center",
            va="center",
            fontsize=14
        )

plt.tight_layout()
plt.savefig(OUT / "02_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()


# =========================================================
# 3. MÉTRICAS DEL BASELINE
# =========================================================

metric_names = [
    "ROC-AUC",
    "PR-AUC",
    "Precision",
    "Recall",
    "F1-score"
]

metric_values = [
    roc_auc,
    pr_auc,
    precision,
    recall,
    f1
]

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(metric_names, metric_values)

ax.set_title("Baseline Logistic Regression - Métricas de VALIDATION")
ax.set_ylabel("Valor")
ax.set_ylim(0, 0.75)

for bar, value in zip(bars, metric_values):
    ax.text(
        bar.get_x() + bar.get_width()/2,
        value,
        f"{value:.4f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig(OUT / "03_baseline_metrics.png", dpi=300, bbox_inches="tight")
plt.close()


# =========================================================
# 4. RESUMEN DE CLASE POSITIVA
# =========================================================

train_rate = train_positive / train_total * 100
val_rate = val_positive / val_total * 100

labels = ["TRAIN", "VALIDATION"]
rates = [train_rate, val_rate]

fig, ax = plt.subplots(figsize=(8, 6))

bars = ax.bar(labels, rates)

ax.set_title("Proporción de TARGET = 1")
ax.set_ylabel("Porcentaje")
ax.set_ylim(0, 10)

for bar, value in zip(bars, rates):
    ax.text(
        bar.get_x() + bar.get_width()/2,
        value,
        f"{value:.2f}%",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig(OUT / "04_positive_class_rate.png", dpi=300, bbox_inches="tight")
plt.close()


print("\nIMÁGENES GENERADAS:")
for file in sorted(OUT.glob("*.png")):
    print(file)
