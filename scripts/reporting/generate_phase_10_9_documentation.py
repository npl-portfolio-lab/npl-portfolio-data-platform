"""Documentación automática de la Fase 10.9."""

import json

from npl_portfolio.core.paths import PROJECT_ROOT, ML_ARTIFACTS_DIR


SOURCE = ML_ARTIFACTS_DIR / "v2/bootstrap_comparison.json"
OUTPUT = PROJECT_ROOT / "docs/ml/fase_10_9_bootstrap_v2.md"


def main():
    data = json.loads(SOURCE.read_text(encoding="utf-8"))

    assert data["phase"] == "10.9"
    assert data["dataset_version"] == "v2"
    assert data["evaluation_partition"] == "validation"
    assert data["test_evaluated"] is False

    rows = []

    for name, result in data["results"].items():
        includes_zero = "Sí" if result["interval_includes_zero"] else "No"

        rows.append(
            f"| {name} | "
            f"{result['observed_difference']:+.6f} | "
            f"[{result['ci_lower']:+.6f}, "
            f"{result['ci_upper']:+.6f}] | "
            f"{includes_zero} |"
        )

    content = f"""# Fase 10.9 — Bootstrap pareado V2

## 1. Objetivo

Estimar la incertidumbre de las diferencias de rendimiento
entre los modelos logísticos balanceado y sin balanceo.

## 2. Datos y metodología

- Partición: VALIDATION V2.
- Registros: {data['validation_rows']:,}.
- Positivos: {data['positive_cases']:,}.
- Variables: {data['features']}.
- Método: {data['method']}.
- Intervalo: {data['interval_method']}.
- Réplicas: {data['iterations']:,}.
- Confianza nominal: {data['confidence_level']:.0%}.
- Semilla: {data['random_state']}.
- TEST: no utilizado.

La diferencia se define como:

Delta = métrica balanceado - métrica sin balanceo.

El remuestreo es pareado porque ambos modelos se
evalúan sobre los mismos índices en cada réplica.

Se conserva la cantidad original de cada clase.

## 3. Resultados

| Métrica | Delta observado | IC 95 % | Incluye cero |
|---|---:|---|---|
{chr(10).join(rows)}

## 4. Interpretación

Los intervalos de ROC-AUC y Precision@K incluyen cero.

El intervalo de PR-AUC queda ligeramente por debajo
de cero, favoreciendo al modelo sin balanceo.

Este resultado debe interpretarse como exploratorio.

No se aplicaron correcciones por comparaciones
múltiples. No se declara superioridad definitiva.

## 5. Limitaciones

- Se utilizaron intervalos percentiles bootstrap.
- No se corrigieron las cinco comparaciones múltiples.
- No se realizó validación temporal independiente.
- TEST V2 permanece reservado.
- TEST V2 no es completamente inédito respecto
  a los experimentos históricos.
- La auditoría temporal de variables sigue pendiente.
- TARGET representa dificultad de pago y no
  recuperación efectiva de cartera castigada.
- El bootstrap cuantifica incertidumbre condicional
  al conjunto VALIDATION y no elimina posibles
  sesgos del diseño de datos.

## 6. Reproducibilidad

Código:

- `src/npl_portfolio/ml/paired_bootstrap.py`
- `scripts/ml/compare_models_bootstrap_v2.py`
- `scripts/ml/audit_bootstrap_v2.py`
- `tests/ml/test_paired_bootstrap.py`

Resultados:

- `data/artifacts/ml/v2/bootstrap_comparison.json`

## 7. Conclusión

No se establece un modelo ganador definitivo.

Las diferencias Top-K son pequeñas y sus intervalos
incluyen cero para las tres capacidades analizadas.

La selección operativa debe considerar también
capacidad, costos, estabilidad y calibración.

Este documento se genera automáticamente desde
el JSON de resultados.
"""

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(content, encoding="utf-8")

    print("=" * 70)
    print("FASE 10.9 - DOCUMENTACION AUTOMATICA")
    print("=" * 70)
    print(f"[OK] Comparaciones: {len(data['results'])}")
    print(f"[OK] Réplicas: {data['iterations']:,}")
    print(f"[OK] Documento: {OUTPUT}")


if __name__ == "__main__":
    main()
