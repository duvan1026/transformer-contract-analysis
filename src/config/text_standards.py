"""
Estándares técnicos para el procesamiento de texto y fragmentación.
Define los requisitos mínimos de calidad y límites físicos del modelo.
"""

# --- LÍMITES DEL MODELO ---
# Tamaño máximo de secuencia para BERT (coherencia entre train e inferencia)
MAX_TOKENS = 512

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

