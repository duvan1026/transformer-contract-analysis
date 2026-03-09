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
# Directorios PRINCIPALES (Nivel 1)
# ==============================

DATA_DIR = BASE_DIR / "data"
MODEL_DATA_DIR = BASE_DIR / "models"
DOCS_DIR = BASE_DIR / "docs"
SRC_DIR  = BASE_DIR / "src"

# ==============================
# GESTIÓN DE DATOS (data/)
# ==============================

RAW_DATA_DIR = DATA_DIR / "raw"
CUAD_JSON = RAW_DATA_DIR / "CUAD_v1.json"
CATEGORY_MAPPING_FILE = RAW_DATA_DIR / "category_mapping.xlsx"

PROCESSED_DATA_DIR = DATA_DIR / "processed"
DATASET_SPANS_CSV = PROCESSED_DATA_DIR / "dataset_spans.csv"
DATASET_TAXONOMY_CSV = PROCESSED_DATA_DIR / "dataset_taxonomy.csv"

SPLITS_DIR = PROCESSED_DATA_DIR / "splits"
TRAIN_DATASET = SPLITS_DIR / "train.csv"
VAL_DATASET = SPLITS_DIR / "validation.csv"
TEST_DATASET = SPLITS_DIR / "test.csv"
SPLIT_REPORT_FILE = SPLITS_DIR / "split_report.txt"

EVALUATION_TEST_DIR = PROCESSED_DATA_DIR / "evaluation"
TEST_DETAILED_RESULT = EVALUATION_TEST_DIR / "test_detailed_results.csv"
CONFUSION_MATRIX = EVALUATION_TEST_DIR / "confusion_matrix.png"
TEST_METRICS_REPORT = EVALUATION_TEST_DIR / "test_metrics_report.txt"

# ==============================
# MODELOS Y ENTRENAMIENTO (models/)
# ==============================

MODEL_BERT_FINAL_DIR = MODEL_DATA_DIR / "bert_final"

MODEL_BERT_BASELINE_DIR = MODEL_DATA_DIR / "bert_baseline"
MODEL_BERT_CHECKPOINT = MODEL_BERT_BASELINE_DIR / "checkpoint-300"

# ==============================
# RESULTADOS Y MÉTRICAS (docs)
# ==============================

DATASET_REPORT_FILE = DOCS_DIR / "dataset_report.txt"


