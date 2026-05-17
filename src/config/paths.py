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

META_DATA_DIR = DATA_DIR / "metadata"
MASTER_DOCUMENT_REGISTRY = META_DATA_DIR / "master_document_registry.csv"

RAW_DATA_DIR = DATA_DIR / "raw"
CUAD_JSON = RAW_DATA_DIR / "CUAD_v1.json"
CATEGORY_MAPPING_FILE = RAW_DATA_DIR / "category_mapping.xlsx"

PROCESSED_DATA_DIR = DATA_DIR / "processed"
DATASET_SPANS_CSV = PROCESSED_DATA_DIR / "dataset_spans.csv"
DATASET_TAXONOMY_CSV = PROCESSED_DATA_DIR / "dataset_taxonomy.csv"

INFERENCE_DIR = PROCESSED_DATA_DIR / "inference"
FRAGMENTS_DATASET = INFERENCE_DIR / "batch_fragments.csv"
REJECTED_AUDIT_DATASET = INFERENCE_DIR / "rejected_fragments_audit.csv"

SPLITS_DIR = PROCESSED_DATA_DIR / "splits"
TRAIN_DATASET = SPLITS_DIR / "train.csv"
VAL_DATASET = SPLITS_DIR / "validation.csv"
TEST_DATASET = SPLITS_DIR / "test.csv"
SPLIT_REPORT_FILE = SPLITS_DIR / "split_report.txt"

# Rutas para los resultados de inferencia (Fase 2)
INFERENCE_SCORED_RESULTS = INFERENCE_DIR / "inference_scored_results.csv"
INFERENCE_METRICS_REPORT = INFERENCE_DIR / "inference_metrics_report.txt"
JSON_OUTPUTS_DIR = INFERENCE_DIR / "extracted_contracts"

# ==============================
# MODELOS Y ENTRENAMIENTO (models/)
# ==============================

MODEL_BERT_FINAL_DIR = MODEL_DATA_DIR / "bert_final"

MODEL_BERT_BASELINE_DIR = MODEL_DATA_DIR / "bert_baseline"
MODEL_BERT_CHECKPOINT = MODEL_BERT_BASELINE_DIR / "checkpoint-300"

# ==============================
# RESULTADOS Y MÉTRICAS (docs)
# ==============================

REPORTS_DIR = DOCS_DIR / "reports"
DATASET_REPORT_FILE = REPORTS_DIR / "dataset_report.txt"
TEST_DETAILED_RESULT = REPORTS_DIR / "test_detailed_results.csv"
CONFUSION_MATRIX = REPORTS_DIR / "confusion_matrix.png"
TEST_METRICS_REPORT = REPORTS_DIR / "test_metrics_report.txt"


