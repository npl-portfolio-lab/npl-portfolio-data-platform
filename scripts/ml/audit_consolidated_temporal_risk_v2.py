"""Fase 10.19: consolidación de auditorías temporales ML V2.

Solo lectura. No modifica datasets ni modelos.
"""

from collections.abc import Mapping


EXPECTED_PHASES = {
    "bureau_balance": "10.13",
    "bureau": "10.14",
    "previous_application": "10.15",
    "pos_cash": "10.16",
    "credit_card": "10.17",
    "installments": "10.18",
}

SOURCE_LABELS = {
    "bureau_balance": "Bureau Balance",
    "bureau": "Bureau",
    "previous_application": "Previous Application",
    "pos_cash": "POS CASH",
    "credit_card": "Credit Card",
    "installments": "Installments",
}


def validate_evidence(evidence: Mapping) -> None:
    """Verifica que estén presentes las seis auditorías."""
    for source, expected_phase in EXPECTED_PHASES.items():
        result = evidence.get(source)

        if not isinstance(result, dict):
            raise ValueError(
                f"Evidencia ausente o inválida: {source}"
            )

        if str(result.get("phase")) != expected_phase:
            raise ValueError(
                f"Fase incorrecta para {source}: "
                f"{result.get('phase')}"
            )


def consolidate(evidence: Mapping) -> dict:
    """Construye una matriz a partir de evidencia real."""
    validate_evidence(evidence)

    rows = []

    for source in EXPECTED_PHASES:
        result = evidence[source]

        rows.append({
            "source": source,
            "label": SOURCE_LABELS[source],
            "phase": result["phase"],
            "classification": result.get(
                "risk_classification",
                "Requiere evaluación punto-en-tiempo",
            ),
            "evidence": result,
        })

    return {
        "phase": "10.19",
        "sources_evaluated": len(rows),
        "risk_matrix": rows,
        "conclusion": (
            "La sensibilidad temporal no demuestra por sí sola "
            "Data Leakage. Se requiere validar la disponibilidad "
            "punto-en-tiempo de las características."
        ),
    }


def collect_evidence() -> dict:
    """Ejecuta las seis auditorías existentes."""
    from scripts.ml.audit_bureau_balance_temporal_sensitivity_v2 import (
        run_audit as bureau_balance,
    )
    from scripts.ml.audit_bureau_temporal_availability_v2 import (
        audit as bureau,
    )
    from scripts.ml.audit_previous_application_temporal_v2 import (
        audit as previous_application,
    )
    from scripts.ml.audit_pos_cash_temporal_sensitivity_v2 import (
        audit as pos_cash,
    )
    from scripts.ml.audit_credit_card_temporal_sensitivity_v2 import (
        audit as credit_card,
    )
    from scripts.ml.audit_installments_temporal_sensitivity_v2 import (
        audit as installments,
    )

    return {
        "bureau_balance": bureau_balance(),
        "bureau": bureau(),
        "previous_application": previous_application(),
        "pos_cash": pos_cash(),
        "credit_card": credit_card(),
        "installments": installments(),
    }


def audit() -> dict:
    """Obtiene y consolida la evidencia de todas las fuentes."""
    return consolidate(collect_evidence())


FEATURE_GROUPS = {
    "bureau": ("BUREAU_", "HAS_BUREAU_HISTORY"),
    "bureau_balance": ("BB_", "HAS_BUREAU_BALANCE_HISTORY"),
    "previous_application": (
        "PREV_", "HAS_PREVIOUS_APPLICATION_HISTORY"
    ),
    "pos_cash": ("POS_", "HAS_POS_HISTORY"),
    "credit_card": ("CC_", "HAS_CREDIT_CARD_HISTORY"),
    "installments": ("INST_", "HAS_INSTALLMENTS_HISTORY"),
}


def get_contract_feature_distribution(contract_path=None):
    """Cuenta variables numéricas por fuente desde el contrato ML V2."""
    from pathlib import Path
    import joblib

    from npl_portfolio.core.paths import PROJECT_ROOT

    path = (
        Path(contract_path)
        if contract_path is not None
        else PROJECT_ROOT
        / "data/artifacts/ml/v2/preprocessing_contract.joblib"
    )

    if not path.is_file():
        raise FileNotFoundError(path)

    # Cargar únicamente el artefacto local confiable del proyecto.
    contract = joblib.load(path)
    numeric_columns = list(contract.numeric_medians)

    distribution = {source: 0 for source in FEATURE_GROUPS}
    distribution["application"] = 0

    for column in numeric_columns:
        matches = [
            source
            for source, (prefix, indicator) in FEATURE_GROUPS.items()
            if column.startswith(prefix) or column == indicator
        ]

        if len(matches) > 1:
            raise ValueError(
                f"Característica asignada a varias fuentes: {column}"
            )

        source = matches[0] if matches else "application"
        distribution[source] += 1

    if sum(distribution.values()) != len(numeric_columns):
        raise ValueError("Distribución de características inconsistente")

    return {
        "total_numeric_features": len(numeric_columns),
        "total_categorical_inputs": len(contract.categorical_modes),
        "distribution": distribution,
    }
