"""
paths.py

Define rutas globales del proyecto para garantizar consistencia y
evitar hardcodear paths en múltiples scripts.
"""

from pathlib import Path


# ==============================
# Directorio raíz del proyecto
# ==============================

BASE_DIR = Path(__file__).resolve().parents[2]


# ==============================
# Directorios de datos
# ==============================

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"


# ==============================
# Archivos específicos
# ==============================

CUAD_JSON = RAW_DATA_DIR / "CUAD_v1.json"

DATASET_SPANS_CSV = PROCESSED_DATA_DIR / "dataset_spans.csv"

CATEGORY_MAPPING_FILE = RAW_DATA_DIR / "category_mapping.xlsx"

DATASET_TAXONOMY_CSV = PROCESSED_DATA_DIR / "dataset_taxonomy.csv"

# ==============================
# Directorios futuros 
# ==============================

# MODELS_DIR = BASE_DIR / "models"
# LOGS_DIR = BASE_DIR / "logs"