"""Documentación automática de evaluación de calibración V2."""

import json

from npl_portfolio.core.paths import PROJECT_ROOT, ML_ARTIFACTS_DIR


SOURCE = ML_ARTIFACTS_DIR / "v2/calibration_evaluation.json"
OUTPUT = PROJECT_ROOT / "docs/ml/fase_10_10_calibracion_v2.md"


def main():
    report = json.loads(SOURCE.read_text(encoding="utf-8"))

    assert report["phase"] == "10.10"
    assert report["dataset_version"] == "v2"
    assert report["test_evaluated"] is False
    assert report["calibration_applied"] is False

    models = report["models"]
    rows = []

    for name in ("unbalanced", "balanced"):
        result = models[name]

        rows.append(
            f"| {name} | "
            f"{result['brier_score']:.6f} | "
            f"{result['log_loss']:.6f} | "
            f"{result['mean_predicted_probability']:.4%} | "
            f"{result['observed_prevalence']:.4%} |"
        )

    bin_sections = []

    for name in ("unbalanced", "balanced"):
        result = models[name]

        lines = [
            f"### Modelo {name}",
            "",
            "| Intervalo | Clientes | Probabilidad media | Frecuencia observada |",
            "|---|---:|---:|---:|",
        ]

        for item in result["bins"]:
            lines.append(
                f"| [{item['lower_bound']:.1f}, "
                f"{item['upper_bound']:.1f}] | "
                f"{item['count']:,} | "
                f"{item['mean_predicted_probability']:.4%} | "
                f"{item['observed_positive_rate']:.4%} |"
            )

        bin_sections.append("\n".join(lines))

    content = f"""# Fase 10.10 — Evaluación de calibración V2

## 1. Objetivo

Evaluar la calidad de las probabilidades originales
de los modelos logísticos V2.

## 2. Datos

- Partición: VALIDATION V2.
- Registros: {report['validation_rows']:,}.
- Casos positivos: {report['positive_cases']:,}.
- Variables: {report['features']}.
- Intervalos de calibración: {report['n_bins']}.
- TEST: no utilizado.

## 3. Metodología

Se evalúan las probabilidades originales, sin aplicar
un calibrador adicional.

Brier Score: error cuadrático medio probabilístico.

Log Loss: pérdida logarítmica probabilística.

Ambas métricas se minimizan.

## 4. Resultados

| Modelo | Brier Score | Log Loss | Probabilidad media | Prevalencia |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

## 5. Análisis por intervalos

{chr(10).join(bin_sections)}

Los intervalos son de ancho uniforme y se omiten
aquellos sin registros.

## 6. Interpretación

El modelo sin balanceo presenta menores valores de
Brier Score y Log Loss en VALIDATION V2.

El modelo balanceado presenta una probabilidad
media considerablemente superior a la prevalencia
observada, indicando sobreestimación agregada.

La proximidad entre probabilidad media y prevalencia
no garantiza calibración adecuada en todos los
intervalos.

## 7. Limitaciones

- Evaluación realizada sobre VALIDATION V2.
- No se ha ajustado ningún calibrador adicional.
- Los resultados son descriptivos.
- La selección previa de modelos utilizó VALIDATION.
- TEST V2 permanece reservado.
- La auditoría temporal de variables sigue pendiente.
- TARGET representa dificultad de pago, no
  recuperación efectiva de cartera castigada.

## 8. Próximo paso

Estudiar métodos de calibración usando datos
separados para ajuste y evaluación.

Cualquier comparación de calibradores debe evitar
evaluarlos sobre los registros empleados para
ajustarlos.

## 9. Reproducibilidad

- `src/npl_portfolio/ml/calibration_evaluator.py`
- `scripts/ml/evaluate_calibration_v2.py`
- `scripts/ml/audit_calibration_v2.py`
- `tests/ml/test_calibration_evaluator.py`
- `data/artifacts/ml/v2/calibration_evaluation.json`

Documento generado automáticamente.
"""

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(content, encoding="utf-8")

    print("=" * 70)
    print("FASE 10.10 - DOCUMENTACION AUTOMATICA")
    print("=" * 70)
    print(f"[OK] Modelos documentados: {len(models)}")
    print(f"[OK] Documento: {OUTPUT}")


if __name__ == "__main__":
    main()
