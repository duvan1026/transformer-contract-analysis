"""
Estándares técnicos para el procesamiento de texto y fragmentación.
Define los requisitos mínimos de calidad y límites físicos del modelo.
"""


# Límite máximo de tokens de entrada al modelo, incluidos [CLS] y [SEP].
MAX_MODEL_TOKENS = 512

# Máximo de tokens de contenido antes de añadir los tokens especiales.
MAX_CONTENT_TOKENS = MAX_MODEL_TOKENS - 2  # 510


# --- REGLAS DE CALIDAD (Filtros de Auditoría) ---
# Longitud mínima de caracteres para que un fragmento no sea considerado "basura"
MIN_CHARS_THRESHOLD = 20

# Activar/Desactivar el filtrado de fragmentos puramente numéricos (ej: "1.1.2")
FILTER_PURE_NUMBERS = True

# --- REPRODUCIBILIDAD ---
# Semilla global para asegurar que la fragmentación y el muestreo sean deterministas
RANDOM_SEED = 42

# --- DELIMITADORES ---
# Delimitador oficial que identifica bloques o párrafos físicos completos
PARAGRAPH_DELIMITER = r'([^\n]+(?:\n[^\n]+)*)'

# Regex para detectar fin de oraciones (usado en subdivisiones)
SENTENCE_BOUNDARY_REGEX = r'[^.!?\n]+[.!?\n]*'

# Caracteres a ignorar al verificar si un texto es solo numérico
NUMERIC_CLEAN_REGEX = r'[.,\s\-\/]'

# Configuración del modelo base
MODEL_NAME = "bert-base-uncased"

# Cantidad de documentos para procesar en un lote de inferencia
BATCH_SIZE_DOCS = 100
EXPERIMENT_NAME_INFERENCE_BASELINE = "v2_inference_Baseline"
EXPERIMENT_NAME_INFERENCE_BALANCED = "v2_inference_Balanced"

N_HOLDOUT = 100
EXPERIMENT_NAME_INFERENCE_BASELINE_HOLDOUT = "v2_inference_Baseline_Holdout"
EXPERIMENT_NAME_INFERENCE_BALANCED_HOLDOUT = "v2_inference_Balanced_Holdout"

# --- FLUJO DE DOCUMENTOS (Registro Maestro) ---
DOC_STATES = {
    "AVAILABLE": "Disponible",      # Contrato crudo listo para ser tomado
    "IN_PROGRESS": "En_proceso",    # Siendo fragmentado por el script 01
    "PROCESSED": "Utilizado",       # Fragmentado con éxito, listo para BERT
    "SCORED": "Clasificado",        # Fase 2 completada (BERT ya calculó scores)
    "NONE": "Ninguno"
}

# --- CONFIGURACIÓN CRÍTICA PARA LA GENERACIÓN DE LA CLASE NONE ---

# Longitud mínima de los candidatos utilizados para generar la clase NONE.
NONE_MIN_CHARS_THRESHOLD = 40

# Densidad alfabética mínima requerida para aceptar un candidato NONE.
NONE_MIN_ALPHA_DENSITY = 0.70

# Expresión regular para aislar únicamente letras puras (usada para calcular la densidad)
ALPHABETIC_CHARS_REGEX = r'[a-zA-Z]'


# --- CONFIGURACIÓN DE INFERENCIA Y EVALUACIÓN (Fase 2) ---

# Umbral de descarte (NONE implícito)
CONFIDENCE_THRESHOLD = 0.75

# Lista de umbrales de confianza a evaluar en la Fase 3 de forma secuencial.
CONFIDENCE_THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

# Número de fragmentos procesados simultáneamente durante la inferencia.
BATCH_SIZE_INFERENCE = 16

# Cuántos candidatos buscar por contrato (Recall@K)
TOP_K_CANDIDATES = 3

# Reglas para la Evaluación del Modelo
OVERLAP_THRESHOLD_CHARS = 30  # Mínimo de caracteres solapados para considerar un "Verdadero Positivo"
RECALL_TOP_K_VALUES = [1, 3]  # Valores de K para medir el Recall a nivel contrato (Top 1 y Top 3)

# --- UMBRALES DE REFERENCIA PARA EL ANÁLISIS DE LONGITUD  ---
# Umbral de palabras usado para detectar fragmentos anómalamente largos.
# Si este valor cambia, el análisis de longitud recalcula automáticamente contra el nuevo umbral.
TOKEN_LENGTH_WORD_THRESHOLD = 200
 
# Rango de tokens (min, max) donde se observó menor acierto del modelo en el análisis de errores.
# Editable: si el rango de menor acierto cambia (por ejemplo, tras reentrenar), actualizar aquí.
TOKEN_LENGTH_RISK_ZONE_TOKENS = (400, 600)

# --- Bins para el análisis de recall por longitud ---
# RECALL_BY_LENGTH_BINS = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]
RECALL_BY_LENGTH_BINS = [0, 50, 100, 150, 250, 400, 600]
 
# Tamaño mínimo de muestra que debe tener un bin para considerarlo confiable
# al determinar automáticamente la zona de riesgo (bin de menor recall).
RECALL_BY_LENGTH_MIN_N_BIN = 15


# ==============================================================================
# REBALANCEO DE SPANS/FRAGMENTOS LARGOS (post train/val/test e inferencia, ver src/utils/rebalancing/).
# ==============================================================================

# Objetivo de reparto (umbral blando): decide en cuántos grupos se divide
# una unidad y qué tan parejos quedan esos grupos entre sí. Una unidad por
# encima de este número NO se corta automáticamente -- solo se corta si,
# además, alguna de sus partes irreducibles supera FRAGMENT_REBALANCE_MAX_TOKENS.
FRAGMENT_REBALANCE_TARGET_TOKENS = 250


# Límite máximo de tokens de CONTENIDO que una pieza puede tener tras el
# rebalanceo, antes de tokenizarse para el modelo. Se deriva de
# MAX_CONTENT_TOKENS (= MAX_MODEL_TOKENS - 2, reservando [CLS]/[SEP]) 
FRAGMENT_REBALANCE_MAX_TOKENS = MAX_CONTENT_TOKENS  # 510
 
# Si es False, los spans de categorías positivas no se dividen durante
# el rebalanceo de entrenamiento para preservar la evidencia que justifica
# su etiqueta.
REBALANCE_INCLUDE_POSITIVE_CATEGORIES = False
 
# Delimitador de la única cláusula de respaldo, usada solo cuando una
# oración completa, por sí sola, ya supera FRAGMENT_REBALANCE_MAX_TOKENS y
# no hay ningún punto, signo de exclamación o interrogación dentro de ella
# para dividirla. Se retiró el nivel adicional que buscaba comas: cortar por
# coma aumenta el riesgo de partir una referencia cruzada o un término legal
# a la mitad, y con la distinción objetivo/límite físico de arriba, este
# nivel de respaldo ya casi nunca hace falta.
CLAUSE_BOUNDARY_SEMICOLON_REGEX = r'[^;]+;?'