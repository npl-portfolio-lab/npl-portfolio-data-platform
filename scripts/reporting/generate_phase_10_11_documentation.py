"""Generación automática de documentación de la Fase 10.11."""

import json

from npl_portfolio.core.paths import PROJECT_ROOT, ML_ARTIFACTS_DIR


SOURCE = (
    ML_ARTIFACTS_DIR
    / "v2/calibration/calibrators_comparison.json"
)

OUTPUT = PROJECT_ROOT / "docs/ml/fase_10_11_recalibracion_v2.md"


def main():
    report = json.loads(SOURCE.read_text(encoding="utf-8"))

    assert report["phase"] == "10.11"
    assert report["dataset_version"] == "v2"
    assert report["test_evaluated"] is False

    rows = []

    for name, methods in report["models"].items():
        for method, metrics in methods.items():
            rows.append(
                f"| {name} | {method} | "
                f"{metrics['brier_score']:.6f} | "
                f"{metrics['log_loss']:.6f} | "
                f"{metrics['roc_auc']:.6f} | "
                f"{metrics['pr_auc']:.6f} |"
            )

    candidates = [
        (name, method, metrics)
        for name, methods in report["models"].items()
        for method, metrics in methods.items()
    ]

    best = min(candidates, key=lambda item: item[2]["brier_score"])

    content = f"""# Fase 10.11 — Recalibración de probabilidades V2

## Objetivo

Comparar probabilidades originales, Sigmoid e Isotonic
para los dos modelos de regresión logística V2.

## Protocolo experimental

- Fuente: VALIDATION V2.
- Separación: estratificada, 50 % FIT y 50 % EVAL.
- Semilla: {report['split']['random_state']}.
- FIT: {report['split']['fit_rows']:,} clientes.
- EVAL: {report['split']['eval_rows']:,} clientes.
- Positivos FIT: {report['split']['fit_positives']:,}.
- Positivos EVAL: {report['split']['eval_positives']:,}.
- Clientes compartidos: {report['split']['overlapping_clients']}.
- TEST V2: no utilizado.

Los modelos originales permanecen sin modificaciones.

Los calibradores se ajustan exclusivamente sobre FIT
y se evalúan exclusivamente sobre EVAL.

## Resultados

| Modelo | Método | Brier ↓ | Log Loss ↓ | ROC-AUC ↑ | PR-AUC ↑ |
|---|---|---:|---:|---:|---:|
{chr(10).join(rows)}

## Interpretación

El menor Brier Score observado corresponde a:

- Modelo: {best[0]}.
- Método: {best[1]}.
- Brier Score: {best[2]['brier_score']:.6f}.

La calibración Sigmoid mejora notablemente las
probabilidades del modelo balanceado.

En el modelo sin balanceo, los métodos adicionales
no mejoran el Brier Score respecto al original.

Isotonic presenta una disminución de PR-AUC
en ambos modelos.

## Decisión provisional

Mantener como candidato principal el modelo
sin balanceo con probabilidades originales.

Conservar el modelo balanceado calibrado mediante
Sigmoid como alternativa experimental.

No se declara superioridad estadística definitiva.

## Limitaciones

- VALIDATION V2 ya se utilizó en análisis anteriores.
- La evaluación es independiente del ajuste de
  los calibradores, pero no de toda selección previa.
- No se realizó comparación estadística pareada
  de las mejoras de calibración.
- La calibración se realizó sobre las probabilidades
  originales, no directamente sobre las puntuaciones
  de decisión del modelo.
- TEST V2 permanece reservado.
- El conjunto TEST V2 no es completamente inédito
  respecto a experimentos históricos.
- La auditoría de posibles fugas temporales
  en las variables sigue pendiente.
- TARGET mide dificultad de pago, no recuperación
  real de cartera castigada.

## Artefactos

- `data/artifacts/ml/v2/calibration/calibrators_comparison.json`
- `data/artifacts/ml/v2/calibration/unbalanced_sigmoid_calibrator.joblib`
- `data/artifacts/ml/v2/calibration/unbalanced_isotonic_calibrator.joblib`
- `data/artifacts/ml/v2/calibration/balanced_sigmoid_calibrator.joblib`
- `data/artifacts/ml/v2/calibration/balanced_isotonic_calibrator.joblib`

## Código

- `src/npl_portfolio/ml/probability_calibrator.py`
- `scripts/ml/train_calibrators_v2.py`
- `scripts/ml/audit_calibrators_v2.py`
- `tests/ml/test_probability_calibrator.py`

Documento generado automáticamente desde el reporte JSON.
"""

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(content, encoding="utf-8")

    print("=" * 70)
    print("FASE 10.11 - DOCUMENTACION AUTOMATICA")
    print("=" * 70)
    print(f"[OK] Alternativas documentadas: {len(candidates)}")
    print(f"[OK] Candidato principal: {best[0]}/{best[1]}")
    print(f"[OK] Documento: {OUTPUT}")


if __name__ == "__main__":
    main()
