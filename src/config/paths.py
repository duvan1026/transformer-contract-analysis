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

MASTER_REGISTRY_BASELINE = META_DATA_DIR / "master_registry_baseline.csv"
MASTER_REGISTRY_BALANCED = META_DATA_DIR / "master_registry_balanced.csv"

RAW_DATA_DIR = DATA_DIR / "raw"
CUAD_JSON = RAW_DATA_DIR / "CUAD_v1.json"
CATEGORY_MAPPING_FILE = RAW_DATA_DIR / "category_mapping.xlsx"

PROCESSED_DATA_DIR = DATA_DIR / "processed"
DATASET_SPANS_CSV = PROCESSED_DATA_DIR / "dataset_spans.csv"
DATASET_TAXONOMY_CSV = PROCESSED_DATA_DIR / "dataset_taxonomy.csv"
DATASET_NONE_MASTER_CSV = PROCESSED_DATA_DIR / "dataset_none_master.csv"
DATASET_WITH_NONE_CSV = PROCESSED_DATA_DIR / "dataset_with_none.csv"

# ---------- Inference -------------------------
INFERENCE_DIR = PROCESSED_DATA_DIR / "inference"

# --- CARRIL 1: INFERENCIA BASELINE ---
INFERENCE_BASELINE_DIR = INFERENCE_DIR / "baseline"
FRAGMENTS_BASELINE = INFERENCE_BASELINE_DIR / "batch_fragments.csv"
REJECTED_AUDIT_BASELINE = INFERENCE_BASELINE_DIR / "rejected_fragments_audit.csv"
SCORED_RESULTS_BASELINE = INFERENCE_BASELINE_DIR / "inference_scored_results.csv"

# --- CARRIL 2: INFERENCIA BALANCEADA ---
INFERENCE_BALANCED_DIR = INFERENCE_DIR / "balanced"
FRAGMENTS_BALANCED = INFERENCE_BALANCED_DIR / "batch_fragments.csv"
REJECTED_AUDIT_BALANCED = INFERENCE_BALANCED_DIR / "rejected_fragments_audit.csv"
SCORED_RESULTS_BALANCED = INFERENCE_BALANCED_DIR / "inference_scored_results.csv"


SPLITS_DIR = PROCESSED_DATA_DIR / "splits"
TRAIN_DATASET = SPLITS_DIR / "train.csv"
VAL_DATASET = SPLITS_DIR / "validation.csv"
TEST_DATASET = SPLITS_DIR / "test.csv"
SPLIT_REPORT_FILE = SPLITS_DIR / "split_report.txt"

# Rutas para los resultados de inferencia 
INFERENCE_SCORED_RESULTS = INFERENCE_DIR / "inference_scored_results.csv"
INFERENCE_METRICS_REPORT = INFERENCE_DIR / "inference_metrics_report.txt"
JSON_OUTPUTS_DIR = INFERENCE_DIR / "extracted_contracts"
INFERENCE_CONFUSION_MATRIX_IMAGE = INFERENCE_DIR / "matriz_confusion_fragmentos.png"
INFERENCE_JSON_DIR_NAME = INFERENCE_DIR / "json_outputs"

# ==============================
# MODELOS Y ENTRENAMIENTO (models/)
# ==============================

MODEL_BERT_FINAL_DIR = MODEL_DATA_DIR / "bert_final"

MODEL_BERT_BASELINE_DIR = MODEL_DATA_DIR / "bert_baseline"
MODEL_BERT_BALANCED_DIR = MODEL_DATA_DIR / "bert_balanced" # Ruta para el modelo con pesos de clase

MODEL_BERT_CHECKPOINT = MODEL_BERT_BASELINE_DIR / "checkpoint-300"

# ==============================
# RESULTADOS Y MÉTRICAS (docs)
# ==============================

# Directorio principal para la generación de evidencias y reportes
REPORTS_DIR = DOCS_DIR / "reports"

# Reporte estructural del dataset (distribución de clases, huérfanos, splits)
DATASET_REPORT_FILE = REPORTS_DIR / "dataset_report.txt"


# --- Evaluación: Modelo Baseline (Control sin pesos de clase) ---

# Auditoría detallada de inferencia (texto original, etiqueta real vs. predicha)
TEST_DETAILED_BASELINE = REPORTS_DIR / "test_detailed_baseline.csv"
# Representación visual de aciertos y dispersión de errores
CONFUSION_MATRIX_BASELINE = REPORTS_DIR / "confusion_matrix_baseline.png"
# Consolidado de métricas clave (Accuracy, Precision, Recall, F1-Score)
TEST_METRICS_REPORT_BASELINE = REPORTS_DIR / "test_metrics_report_baseline.txt"


# --- Evaluación: Modelo Balanceado (Propuesta con pesos de clase) ---

# Auditoría detallada de inferencia (texto original, etiqueta real vs. predicha)
TEST_DETAILED_BALANCED = REPORTS_DIR / "test_detailed_balanced.csv"
# Representación visual de aciertos y dispersión de errores
CONFUSION_MATRIX_BALANCED = REPORTS_DIR / "confusion_matrix_balanced.png"
# Consolidado de métricas clave del modelo propuesto
TEST_METRICS_REPORT_BALANCED = REPORTS_DIR / "test_metrics_report_balanced.txt"

