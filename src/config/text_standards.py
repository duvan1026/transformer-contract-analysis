"""
Estándares técnicos para el procesamiento de texto y fragmentación.
Define los requisitos mínimos de calidad y límites físicos del modelo.
"""


# --- LÍMITES DEL MODELO ---
# Límite físico de entrada de BERT: incluye los tokens especiales [CLS] y
# [SEP] añadidos por el tokenizer (add_special_tokens=True, comportamiento
# por defecto de HuggingFace). Es el valor usado como max_length en la
# tokenización final real: ContractDataset (entrenamiento, bert_dataset.py,
# literal hardcodeado que NO se toca en este cambio) y BERTInferenceDataset
# (inferencia, inference_dataset.py).
MAX_MODEL_TOKENS = 512

# Máximo de tokens de CONTENIDO puro (sin tokens especiales). Se deriva de
# MAX_MODEL_TOKENS reservando 2 posiciones para [CLS] y [SEP], de modo que
# ambos valores no puedan desincronizarse manualmente:
#   MAX_CONTENT_TOKENS + [CLS] + [SEP] == MAX_MODEL_TOKENS
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

########### Warning ###################
# To Do: MODEL_NAME = "bert-base-uncased" tanto aquí como en label_mapping.py. No es un error grave, pero te sugiero borrarlo de uno de los dos lados para mantener la "Única Fuente de Verdad" (recomiendo dejarlo en label_mapping.py o cambiarle el nombre aquí a TOKENIZER_CHECKPOINT si lo quieres separar).
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

# Tamaño mínimo en caracteres para simular un párrafo denso
# Alrededor de 40-50 palabras para evitar el "atajo de longitud" (shortcut Learning)
NONE_MIN_CHARS_THRESHOLD = 40

# Densidad alfabética mínima requerida
# Basado en los estándares de limpieza de Gopher (DeepMind) y ROOTS (BigScience)
# 0.70 significa que el 70% del párrafo deben ser letras de la A a la Z
NONE_MIN_ALPHA_DENSITY = 0.70

# Expresión regular para aislar únicamente letras puras (usada para calcular la densidad)
ALPHABETIC_CHARS_REGEX = r'[a-zA-Z]'


# --- CONFIGURACIÓN DE INFERENCIA Y EVALUACIÓN (Fase 2) ---

# Umbral de descarte (NONE implícito)
CONFIDENCE_THRESHOLD = 0.75

# Lista de umbrales de confianza a evaluar en la Fase 3 de forma secuencial.
CONFIDENCE_THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

# Cantidad de fragmentos de texto que BERT procesará en paralelo por cada iteración.
# Un valor de 16 optimiza el uso de la GPU (VRAM) sin riesgo de saturar la memoria (Out of Memory).
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
 
# Si es False (valor por defecto), las categorías positivas (todo lo que no
# sea "NONE") nunca se dividen en el rebalanceo de entrenamiento -- se dejan
# exactamente como las extrajo extract_spans.py desde CUAD. Dividir un span
# ya etiquetado arriesga que el trozo resultante ya no contenga el elemento
# que justifica esa etiqueta (por ejemplo, el verbo de prohibición de una
# cláusula de no competencia puede quedar en un trozo y la enumeración de
# casos en otro, sin ese verbo). Cambiar a True revierte al comportamiento
# anterior (todas las filas se dividen por igual, sin distinguir categoría).
REBALANCE_INCLUDE_POSITIVE_CATEGORIES = False
 
# Delimitador de la única cláusula de respaldo, usada solo cuando una
# oración completa, por sí sola, ya supera FRAGMENT_REBALANCE_MAX_TOKENS y
# no hay ningún punto, signo de exclamación o interrogación dentro de ella
# para dividirla. Se retiró el nivel adicional que buscaba comas: cortar por
# coma aumenta el riesgo de partir una referencia cruzada o un término legal
# a la mitad, y con la distinción objetivo/límite físico de arriba, este
# nivel de respaldo ya casi nunca hace falta.
CLAUSE_BOUNDARY_SEMICOLON_REGEX = r'[^;]+;?'