"""FASE 10.2 - Análisis de capacidad operativa sin reentrenamiento."""

import json

from npl_portfolio.core.paths import ML_ARTIFACTS_DIR


INPUT_PATH = ML_ARTIFACTS_DIR / "advanced_threshold_evaluation.json"
OUTPUT_PATH = ML_ARTIFACTS_DIR / "operational_capacity_analysis.json"

CAPACITIES = [1000, 3000, 5000]


def select_best_configuration(models, capacity):
    candidates = []

    for model_name, model_data in models.items():
        for result in model_data["results"]:
            alerts = int(result["predicted_positives"])

            if alerts > capacity:
                continue

            candidates.append({
                "model": model_name,
                "threshold": float(result["threshold"]),
                "alerts": alerts,
                "tp": int(result["tp"]),
                "fp": int(result["fp"]),
                "fn": int(result["fn"]),
                "tn": int(result["tn"]),
                "precision": float(result["precision"]),
                "recall": float(result["recall"]),
                "f1": float(result["f1"]),
                "capacity_utilization": alerts / capacity,
            })

    if not candidates:
        return None

    # Prioridad: más positivos detectados.
    # Desempate: menos falsos positivos y menos alertas.
    return max(
        candidates,
        key=lambda item: (
            item["tp"],
            -item["fp"],
            -item["alerts"],
        ),
    )


def main():
    print("=" * 70)
    print("FASE 10.2 - ANALISIS DE CAPACIDAD OPERATIVA")
    print("=" * 70)

    if not INPUT_PATH.is_file():
        raise FileNotFoundError(INPUT_PATH)

    with INPUT_PATH.open(encoding="utf-8") as file:
        source = json.load(file)

    if source.get("phase") != "10.1":
        raise ValueError("El reporte no corresponde a la Fase 10.1")

    models = source["models"]

    expected_models = {"unbalanced", "balanced"}

    if set(models) != expected_models:
        raise ValueError(
            "Se esperaban exactamente los modelos balanced y unbalanced"
        )

    report = {
        "phase": "10.2",
        "source": INPUT_PATH.name,
        "dataset": source["dataset"],
        "n_samples": source["n_samples"],
        "positive_samples": source["positive_samples"],
        "negative_samples": source["negative_samples"],
        "selection_criterion": (
            "maximize_true_positives_under_alert_capacity"
        ),
        "capacity_unit": "alerts_per_validation_dataset",
        "scenarios": [],
    }

    print(f"\nDataset: {source['dataset']}")
    print(f"Registros: {source['n_samples']:,}")
    print(f"Positivos reales: {source['positive_samples']:,}")

    for capacity in CAPACITIES:
        selected = select_best_configuration(models, capacity)

        scenario = {
            "maximum_alerts": capacity,
            "selected": selected,
        }

        report["scenarios"].append(scenario)

        print("\n" + "-" * 70)
        print(f"CAPACIDAD MAXIMA: {capacity:,} ALERTAS")

        if selected is None:
            print("[WARNING] Ninguna configuracion cumple la capacidad")
            continue

        print(f"Modelo: {selected['model']}")
        print(f"Threshold: {selected['threshold']:.2f}")
        print(f"Alertas: {selected['alerts']:,}")
        print(f"Positivos detectados (TP): {selected['tp']:,}")
        print(f"Falsos positivos (FP): {selected['fp']:,}")
        print(f"Positivos no detectados (FN): {selected['fn']:,}")
        print(f"Precision: {selected['precision']:.4f}")
        print(f"Recall: {selected['recall']:.4f}")
        print(f"F1: {selected['f1']:.4f}")
        print(
            "Uso de capacidad: "
            f"{selected['capacity_utilization']:.2%}"
        )

    # Escribir de forma atomica para evitar reportes parciales.
    temporary_path = OUTPUT_PATH.with_suffix(".json.tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )

    temporary_path.replace(OUTPUT_PATH)

    print(f"\n[OK] Reporte generado: {OUTPUT_PATH}")
    print("[OK] FASE 10.2 FINALIZADA")


if __name__ == "__main__":
    main()
