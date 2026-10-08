import pytest

from scripts.reporting.generate_phase_10_12_documentation import (
    parse_integrity,
    parse_temporal,
)


def test_temporal_rejects_incomplete_audit():
    output = (
        "[PASS] bureau.DAYS_CREDIT: "
        "min=-2922 max=0 futuros=0"
    )

    with pytest.raises(ValueError, match="7 controles"):
        parse_temporal(output)


def test_temporal_rejects_future_dates():
    output = "\n".join(
        f"[PASS] source_{i}.DAYS: min=-10 max=1 futuros=1"
        for i in range(7)
    )

    with pytest.raises(ValueError, match="fechas futuras"):
        parse_temporal(output)


def test_integrity_rejects_incomplete_audit():
    output = (
        "[PASS] TRAIN: 307,511 clientes unicos\n"
        "Columnas verificadas: 121"
    )

    with pytest.raises(ValueError, match="incompleta"):
        parse_integrity(output)


def test_integrity_rejects_inconsistent_feature_count():
    datasets = (
        "[PASS] TRAIN: 307,511 clientes unicos\n"
        "[PASS] TEST: 48,744 clientes unicos\n"
    )

    sources = "\n".join(
        (
            f"[PASS] source_{i}: filas=100, features=2, "
            "con_historial=80, sin_historial=20, diferencias=0"
        )
        for i in range(6)
    )

    output = (
        datasets
        + sources
        + "\nColumnas verificadas: 121"
    )

    with pytest.raises(ValueError, match="inconsistente"):
        parse_integrity(output)
