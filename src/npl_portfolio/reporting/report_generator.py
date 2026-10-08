from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ReportGenerator:
    """Genera informes Markdown legibles desde resultados registrados."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)

    def generate(
        self,
        phase: str,
        title: str,
        results: dict[str, Any],
        filename: str,
    ) -> Path:
        if not results:
            raise ValueError("No hay resultados para documentar.")

        if Path(filename).name != filename or not filename.endswith(".md"):
            raise ValueError("El nombre debe ser un archivo .md válido.")

        metrics = results.get("metrics", {})
        required = (
            "roc_auc",
            "pr_auc",
            "precision",
            "recall",
            "f1",
            "tn",
            "fp",
            "fn",
            "tp",
            "threshold",
        )
        if any(key not in metrics for key in required):
            raise ValueError("Faltan métricas necesarias para el informe ML.")

        def percent(value: float) -> str:
            return f"{value * 100:.2f}%"

        def integer(value: int) -> str:
            return f"{value:,}"

        precision = metrics["precision"]
        recall = metrics["recall"]
        tp = metrics["tp"]
        fp = metrics["fp"]
        fn = metrics["fn"]
        tn = metrics["tn"]

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        lines = [
            f"# Fase {phase} — {title}",
            "",
            f"**Informe generado:** {timestamp}",
            "",
            "## 1. Objetivo",
            "",
            "Evaluar el desempeño de un modelo de clasificación "
            "para identificar clientes con dificultades de pago "
            "(`TARGET = 1`).",
            "",
            "## 2. Configuración del experimento",
            "",
            "| Parámetro | Valor |",
            "|---|---|",
            f"| Modelo | {results.get('model', 'No registrado')} |",
            f"| Dataset evaluado | {results.get('dataset', 'No registrado')} |",
            f"| Registros evaluados | {integer(results['n_samples'])} |",
            f"| Variables | {integer(results['n_features'])} |",
            f"| Escalador | {results.get('scaler', 'No registrado')} |",
            f"| Solver | {results.get('solver', 'No registrado')} |",
            f"| Balanceo de clases | {results.get('class_weight', 'No registrado')} |",
            f"| Iteraciones | {results.get('iterations', 'No registrado')} |",
            f"| Umbral de clasificación | {metrics['threshold']} |",
            "",
            "## 3. Resultados del entrenamiento",
            "",
            f"El entrenamiento registró una duración de "
            f"**{results.get('training_seconds', 0):.2f} segundos**.",
            "",
            f"**Advertencia de convergencia:** "
            f"{'Sí' if results.get('convergence_warning') else 'No'}",
            "",
            "## 4. Métricas de evaluación",
            "",
            "| Métrica | Resultado | ¿Qué mide? |",
            "|---|---:|---|",
            f"| ROC-AUC | {percent(metrics['roc_auc'])} | "
            "Capacidad de distinguir entre clases |",
            f"| PR-AUC | {percent(metrics['pr_auc'])} | "
            "Precisión y cobertura de la clase positiva |",
            f"| Precision | {percent(precision)} | "
            "Proporción de predicciones positivas correctas |",
            f"| Recall | {percent(recall)} | "
            "Proporción de positivos reales detectados |",
            f"| F1-score | {percent(metrics['f1'])} | "
            "Equilibrio entre precision y recall |",
            "",
            "## 5. Matriz de confusión",
            "",
            "| Resultado | Cantidad | Interpretación |",
            "|---|---:|---|",
            f"| TN | {integer(tn)} | Negativos correctamente identificados |",
            f"| FP | {integer(fp)} | Negativos clasificados como positivos |",
            f"| FN | {integer(fn)} | Positivos que el modelo no detectó |",
            f"| TP | {integer(tp)} | Positivos correctamente identificados |",
            "",
            "## 6. Interpretación de los resultados",
            "",
            f"El modelo identificó correctamente **{integer(tp)}** "
            f"de **{integer(tp + fn)}** casos positivos reales, "
            f"equivalentes a un recall de **{percent(recall)}**.",
            "",
            f"De los **{integer(tp + fp)}** casos clasificados "
            f"como positivos, **{integer(tp)}** fueron correctos. "
            f"Esto representa una precision de **{percent(precision)}**.",
            "",
            f"Se registraron **{integer(fp)} falsos positivos** "
            f"y **{integer(fn)} falsos negativos**.",
            "",
            "## 7. Conclusiones y limitaciones",
            "",
            "El modelo permite identificar una proporción de los "
            "casos positivos, pero también genera errores de "
            "clasificación que deben evaluarse según su impacto "
            "en el negocio.",
            "",
            "Estas métricas corresponden al conjunto de validación. "
            "No constituyen una evaluación sobre datos futuros "
            "ni garantizan el mismo desempeño en producción.",
            "",
            "La selección del modelo requiere comparar alternativas "
            "bajo condiciones equivalentes y analizar el umbral "
            "de clasificación.",
            "",
        ]

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / filename
        temporary_path = output_path.with_suffix(".md.tmp")

        temporary_path.write_text(
            "\n".join(lines),
            encoding="utf-8",
        )
        temporary_path.replace(output_path)

        return output_path
