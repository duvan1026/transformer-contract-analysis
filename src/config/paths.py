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
GROUND_TRUTH_BASELINE = INFERENCE_BASELINE_DIR / "ground_truth_results.csv"

# --- CARRIL 2: INFERENCIA BALANCEADA ---
INFERENCE_BALANCED_DIR = INFERENCE_DIR / "balanced"
FRAGMENTS_BALANCED = INFERENCE_BALANCED_DIR / "batch_fragments.csv"
REJECTED_AUDIT_BALANCED = INFERENCE_BALANCED_DIR / "rejected_fragments_audit.csv"
SCORED_RESULTS_BALANCED = INFERENCE_BALANCED_DIR / "inference_scored_results.csv"
GROUND_TRUTH_BALANCED = INFERENCE_BALANCED_DIR / "ground_truth_results.csv"

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

REPORT_FULL_CONFIDENCE_NAME = "full_confidence_report"
REPORT_THRESHOLD_PREFIX = "threshold_"

# ==============================
# MODELOS Y ENTRENAMIENTO (models/)
# ==============================

MODEL_BERT_BASELINE_DIR = MODEL_DATA_DIR / "bert_baseline"
MODEL_BERT_BALANCED_DIR = MODEL_DATA_DIR / "bert_balanced"

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

# ==============================
# ANÁLISIS DE LONGITUD DE TOKENS (docs/reports/token_length_analysis/)
# ==============================
# Análisis diagnóstico (no productivo): compara la longitud de spans en
# train/val/test contra la longitud real observada en inferencia (ground truth),
# por clase, para evaluar si la fragmentación por párrafo (H04/H05/H06) amerita
# rediseño (F02/F03). Se ejecuta manualmente via src/analysis/05_analyze_token_length.py

TOKEN_LENGTH_DIR = REPORTS_DIR / "token_length_analysis"

# --- CARRIL 1: BASELINE ---
TOKEN_LENGTH_BASELINE_DIR = TOKEN_LENGTH_DIR / "baseline"
TOKEN_LENGTH_STATS_BASELINE = TOKEN_LENGTH_BASELINE_DIR / "token_length_stats.xlsx"
TOKEN_LENGTH_BOXPLOT_BASELINE = TOKEN_LENGTH_BASELINE_DIR / "boxplot_tokens_por_clase.png"
TOKEN_LENGTH_ECDF_BASELINE = TOKEN_LENGTH_BASELINE_DIR / "ecdf_train_vs_inferencia.png"
TOKEN_LENGTH_LABEL_AUDIT_BASELINE = TOKEN_LENGTH_BASELINE_DIR / "etiquetas_no_reconocidas.csv"

# Volumen bruto de fragmentos por clase y fuente (escala log)
TOKEN_LENGTH_CLASS_BALANCE_BASELINE = TOKEN_LENGTH_BASELINE_DIR / "balance_clases.png"
# Composicion relativa por fuente, incluyendo NONE
TOKEN_LENGTH_COMPOSITION_WITH_NONE_BASELINE = TOKEN_LENGTH_BASELINE_DIR / "composicion_relativa_con_none.png"
# Composicion relativa por fuente, solo clases positivas (NONE excluido y renormalizado)
TOKEN_LENGTH_COMPOSITION_POSITIVE_ONLY_BASELINE = TOKEN_LENGTH_BASELINE_DIR / "composicion_relativa_solo_positivas.png"
# Spans de train/val/test en riesgo de truncamiento (>MAX_TOKENS)
TOKEN_LENGTH_TRUNCATION_RISK_BASELINE = TOKEN_LENGTH_BASELINE_DIR / "riesgo_truncamiento.png"
# Ranking por clase del % de fragmentos reales en zona de riesgo (solo inferencia)
TOKEN_LENGTH_RISK_BY_CLASS_BASELINE = TOKEN_LENGTH_BASELINE_DIR / "ranking_riesgo_por_clase.png"


# --- CARRIL 2: BALANCED ---
TOKEN_LENGTH_BALANCED_DIR = TOKEN_LENGTH_DIR / "balanced"
TOKEN_LENGTH_STATS_BALANCED = TOKEN_LENGTH_BALANCED_DIR / "token_length_stats.xlsx"
TOKEN_LENGTH_BOXPLOT_BALANCED = TOKEN_LENGTH_BALANCED_DIR / "boxplot_tokens_por_clase.png"
TOKEN_LENGTH_ECDF_BALANCED = TOKEN_LENGTH_BALANCED_DIR / "ecdf_train_vs_inferencia.png"
TOKEN_LENGTH_LABEL_AUDIT_BALANCED = TOKEN_LENGTH_BALANCED_DIR / "etiquetas_no_reconocidas.csv"

TOKEN_LENGTH_CLASS_BALANCE_BALANCED = TOKEN_LENGTH_BALANCED_DIR / "balance_clases.png"
TOKEN_LENGTH_COMPOSITION_WITH_NONE_BALANCED = TOKEN_LENGTH_BALANCED_DIR / "composicion_relativa_con_none.png"
TOKEN_LENGTH_COMPOSITION_POSITIVE_ONLY_BALANCED = TOKEN_LENGTH_BALANCED_DIR / "composicion_relativa_solo_positivas.png"
TOKEN_LENGTH_TRUNCATION_RISK_BALANCED = TOKEN_LENGTH_BALANCED_DIR / "riesgo_truncamiento.png"
TOKEN_LENGTH_RISK_BY_CLASS_BALANCED = TOKEN_LENGTH_BALANCED_DIR / "ranking_riesgo_por_clase.png"


# ==============================
# ANÁLISIS DE RECALL POR LONGITUD
# ==============================
# Verifica, con el ground truth real, si la zona de menor acierto del
# modelo sigue vigente, en vez de asumirla como un valor fijo. Se ejecuta ANTES de analyze_token_length.py:
# su resultado (zona de riesgo calculada) alimenta las gráficas de longitud.

RECALL_BY_LENGTH_DIR = REPORTS_DIR / "recall_by_length_analysis"

# --- CARRIL 1: BASELINE ---
RECALL_BY_LENGTH_BASELINE_DIR = RECALL_BY_LENGTH_DIR / "baseline"
RECALL_BY_LENGTH_STATS_BASELINE = RECALL_BY_LENGTH_BASELINE_DIR / "recall_por_longitud.xlsx"
RECALL_BY_LENGTH_CHART_BASELINE = RECALL_BY_LENGTH_BASELINE_DIR / "recall_por_longitud.png"
# Archivo generado por analyze_recall_by_length.py y leido por analyze_token_length.py
RECALL_BY_LENGTH_RISK_ZONE_FILE_BASELINE = RECALL_BY_LENGTH_BASELINE_DIR / "zona_riesgo_calculada.json"

# --- CARRIL 2: BALANCED ---
RECALL_BY_LENGTH_BALANCED_DIR = RECALL_BY_LENGTH_DIR / "balanced"
RECALL_BY_LENGTH_STATS_BALANCED = RECALL_BY_LENGTH_BALANCED_DIR / "recall_por_longitud.xlsx"
RECALL_BY_LENGTH_CHART_BALANCED = RECALL_BY_LENGTH_BALANCED_DIR / "recall_por_longitud.png"
RECALL_BY_LENGTH_RISK_ZONE_FILE_BALANCED = RECALL_BY_LENGTH_BALANCED_DIR / "zona_riesgo_calculada.json"

# --- Simulacion de impacto: escenario actual vs. objetivo (longitud igualada a train) ---
IMPACT_SIMULATION_STATS_BASELINE = RECALL_BY_LENGTH_BASELINE_DIR / "simulacion_impacto.xlsx"
IMPACT_SIMULATION_CHART_BASELINE = RECALL_BY_LENGTH_BASELINE_DIR / "simulacion_impacto.png"

IMPACT_SIMULATION_STATS_BALANCED = RECALL_BY_LENGTH_BALANCED_DIR / "simulacion_impacto.xlsx"
IMPACT_SIMULATION_CHART_BALANCED = RECALL_BY_LENGTH_BALANCED_DIR / "simulacion_impacto.png"

ANNOTATION_REGISTRY_BASELINE = INFERENCE_BASELINE_DIR / "annotation_registry.csv"
ANNOTATION_REGISTRY_BALANCED = INFERENCE_BALANCED_DIR / "annotation_registry.csv"

# ==============================================================================
# REBALANCEO DE SPANS/FRAGMENTOS LARGOS
# ==============================================================================

REBALANCING_REPORTS_DIR = REPORTS_DIR / "rebalancing"

# --- Train/val/test (un único CSV de rechazos y un único reporte que
# cubre los tres splits, con columna/sección "split", igual que ya hace
# SPLIT_REPORT_FILE con train/val/test combinados) ---
REBALANCE_REJECTED_TRAIN = PROCESSED_DATA_DIR / "rebalance_rejected_train_val_test.csv"
REBALANCE_REPORT_TRAIN = REBALANCING_REPORTS_DIR / "rebalance_train_val_test_report.txt"

# --- Inferencia: un archivo de rechazos y un reporte por carril,
# consistente con el resto de artefactos de inferencia (aislados por
# experimento en INFERENCE_BASELINE_DIR / INFERENCE_BALANCED_DIR) ---
REBALANCE_REJECTED_INFERENCE_BASELINE = INFERENCE_BASELINE_DIR / "rebalance_rejected_fragments.csv"
REBALANCE_REPORT_INFERENCE_BASELINE = REBALANCING_REPORTS_DIR / "baseline_report.txt"

REBALANCE_REJECTED_INFERENCE_BALANCED = INFERENCE_BALANCED_DIR / "rebalance_rejected_fragments.csv"
REBALANCE_REPORT_INFERENCE_BALANCED = REBALANCING_REPORTS_DIR / "balanced_report.txt"
