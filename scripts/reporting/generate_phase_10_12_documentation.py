"""Genera documentación de la Fase 10.12 desde las auditorías.

No modifica datasets, modelos ni artefactos ML.
"""

import argparse
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs/ml/fase_10_12_auditoria_temporal_v2.md"

TEMPORAL = ROOT / "scripts/ml/audit_temporal_leakage_v2.py"
INTEGRITY = ROOT / "scripts/ml/audit_historical_feature_integrity_v2.py"


def run_auditor(path: Path) -> str:
    result = subprocess.run(
        [sys.executable, str(path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Auditoría fallida: {path.name}\n"
            f"{result.stdout}\n{result.stderr}"
        )

    return result.stdout


def parse_temporal(output: str):
    pattern = re.compile(
        r"^\[PASS\] (\S+): min=([-\d.]+) "
        r"max=([-\d.]+) futuros=([\d,]+)$",
        re.MULTILINE,
    )

    rows = pattern.findall(output)

    if len(rows) != 7:
        raise ValueError(
            f"Se esperaban 7 controles temporales; encontrados: {len(rows)}"
        )

    if any(int(future.replace(",", "")) != 0 for _, _, _, future in rows):
        raise ValueError("Se detectaron fechas futuras.")

    warnings = re.findall(
        r"^\[WARN\] (.+)$",
        output,
        re.MULTILINE,
    )

    return rows, warnings


def parse_integrity(output: str):
    dataset_pattern = re.compile(
        r"^\[PASS\] (TRAIN|TEST): ([\d,]+) clientes unicos$",
        re.MULTILINE,
    )

    source_pattern = re.compile(
        r"^\[PASS\] (\w+): filas=([\d,]+), "
        r"features=(\d+), con_historial=([\d,]+), "
        r"sin_historial=([\d,]+), diferencias=0$",
        re.MULTILINE,
    )

    datasets = dataset_pattern.findall(output)
    sources = source_pattern.findall(output)

    if len(datasets) != 2 or len(sources) != 6:
        raise ValueError(
            "Salida de integridad incompleta: "
            f"datasets={len(datasets)}, fuentes={len(sources)}"
        )

    total_match = re.search(
        r"Columnas verificadas:\s*(\d+)",
        output,
    )

    if not total_match:
        raise ValueError("No se encontró el total de columnas.")

    total = int(total_match.group(1))
    calculated = sum(int(row[2]) for row in sources)

    if total != calculated:
        raise ValueError(
            f"Total de columnas inconsistente: {total} != {calculated}"
        )

    return datasets, sources, total


def generate_markdown(
    temporal_rows,
    warnings,
    datasets,
    sources,
    total_features,
) -> str:
    timestamp = datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )

    lines = [
        "# Fase 10.12 — Auditoría temporal e integridad histórica V2",
        "",
        "> Documento generado automáticamente. No editar manualmente.",
        "",
        f"**Fecha de generación:** {timestamp}",
        "",
        "## 1. Objetivo",
        "",
        "Verificar integridad histórica, correspondencia de características "
        "y controles temporales de los datasets V2.",
        "",
        "La auditoría es de solo lectura y no reentrena modelos.",
        "",
        "## 2. Integridad de datasets",
        "",
        "| Dataset | Clientes únicos | Estado |",
        "|---|---:|---|",
    ]

    for name, count in datasets:
        lines.append(f"| {name} | {count} | PASS |")

    lines.extend([
        "",
        "## 3. Integridad de características históricas",
        "",
        "| Fuente | Filas | Características | Con historial | "
        "Sin historial | Estado |",
        "|---|---:|---:|---:|---:|---|",
    ])

    for name, rows, features, matched, missing in sources:
        lines.append(
            f"| {name} | {rows} | {features} | "
            f"{matched} | {missing} | PASS |"
        )

    lines.extend([
        "",
        f"**Total de características verificadas:** {total_features}",
        "",
        "Las características se compararon para clientes presentes "
        "en ambas tablas. También se comprobaron las banderas "
        "de historial y la unicidad de las claves.",
        "",
        "## 4. Auditoría temporal",
        "",
        "| Fuente y columna | Mínimo | Máximo | Valores futuros |",
        "|---|---:|---:|---:|",
    ])

    for name, minimum, maximum, future in temporal_rows:
        lines.append(
            f"| {name} | {minimum} | {maximum} | {future} |"
        )

    lines.extend([
        "",
        "## 5. Advertencias",
        "",
    ])

    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("No se reportaron advertencias.")

    lines.extend([
        "",
        "## 6. Limitaciones",
        "",
        "- La ausencia de fechas positivas no demuestra ausencia "
        "completa de fuga temporal.",
        "- La disponibilidad de cada variable en el instante "
        "de predicción no está completamente demostrada.",
        "- Los constructores históricos examinados no tienen "
        "filtros temporales explícitos.",
        "- La integridad de las características no demuestra "
        "por sí sola reproducibilidad completa desde las fuentes.",
        "- Esta auditoría no evalúa métricas predictivas sobre TEST.",
        "",
        "## 7. Conclusión",
        "",
        "Los controles ejecutados no detectaron inconsistencias "
        "en las características históricas verificadas ni fechas "
        "positivas en las siete columnas temporales auditadas.",
        "",
        "La ausencia completa de *data leakage* permanece "
        "sin certificar debido a las limitaciones temporales "
        "documentadas.",
        "",
        "## 8. Reproducción",
        "",
        "```bash",
        "python scripts/ml/audit_temporal_leakage_v2.py",
        "python scripts/ml/audit_historical_feature_integrity_v2.py",
        "python scripts/reporting/generate_phase_10_12_documentation.py "
        "--overwrite",
        "python -m pytest -q",
        "```",
        "",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Autoriza reemplazar el informe existente.",
    )
    args = parser.parse_args()

    if OUTPUT.exists() and not args.overwrite:
        parser.error(
            f"El informe ya existe: {OUTPUT}. "
            "Usa --overwrite para reemplazarlo."
        )

    print("Ejecutando auditoría temporal...")
    temporal_output = run_auditor(TEMPORAL)

    print("Ejecutando auditoría de integridad...")
    integrity_output = run_auditor(INTEGRITY)

    temporal_rows, warnings = parse_temporal(temporal_output)
    datasets, sources, total = parse_integrity(integrity_output)

    markdown = generate_markdown(
        temporal_rows,
        warnings,
        datasets,
        sources,
        total,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(markdown, encoding="utf-8")

    print(f"[PASS] Informe generado: {OUTPUT}")
    print(f"[PASS] Controles temporales: {len(temporal_rows)}")
    print(f"[PASS] Fuentes históricas: {len(sources)}")
    print(f"[PASS] Características: {total}")
    print(f"[WARN] Advertencias documentadas: {len(warnings)}")


if __name__ == "__main__":
    main()
