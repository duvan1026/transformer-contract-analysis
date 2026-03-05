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
DOCS_DIR = BASE_DIR / "docs"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

SPLITS_DIR = PROCESSED_DATA_DIR / "splits"

# ==============================
# Archivos específicos
# ==============================

CUAD_JSON = RAW_DATA_DIR / "CUAD_v1.json"

DATASET_SPANS_CSV = PROCESSED_DATA_DIR / "dataset_spans.csv"

CATEGORY_MAPPING_FILE = RAW_DATA_DIR / "category_mapping.xlsx"

DATASET_TAXONOMY_CSV = PROCESSED_DATA_DIR / "dataset_taxonomy.csv"

DATASET_REPORT_FILE = DOCS_DIR / "dataset_report.txt"


TRAIN_DATASET = SPLITS_DIR / "train.csv"

VAL_DATASET = SPLITS_DIR / "validation.csv"

TEST_DATASET = SPLITS_DIR / "test.csv"

SPLIT_REPORT_FILE = SPLITS_DIR / "split_report.txt"

# ==============================
# Directorios futuros 
# ==============================

# MODELS_DIR = BASE_DIR / "models"
# LOGS_DIR = BASE_DIR / "logs"