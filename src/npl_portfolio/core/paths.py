from pathlib import Path

# Raíz del repositorio
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Directorios principales
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = PROJECT_ROOT / "docs"
CONFIG_DIR = PROJECT_ROOT / "config"

# Artefactos
ML_ARTIFACTS_DIR = DATA_DIR / "artifacts" / "ml"
VISUAL_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Datos procesados
PROCESSED_DIR = DATA_DIR / "processed"
ML_DATA_DIR = PROCESSED_DIR / "ml"
