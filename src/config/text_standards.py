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
# Define qué considera el sistema como un salto de párrafo natural
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
EXPERIMENT_NAME = "v2_inferencia_100_docs"

# --- FLUJO DE DOCUMENTOS (Registro Maestro) ---
DOC_STATES = {
    "AVAILABLE": "Disponible",      # Contrato crudo listo para ser tomado
    "IN_PROGRESS": "En_proceso",    # Siendo fragmentado por el script 01
    "PROCESSED": "Utilizado",       # Fragmentado con éxito, listo para BERT
    "SCORED": "Clasificado",        # Fase 2 completada (BERT ya calculó scores)
    "NONE": "Ninguno"
}

# --- CONFIGURACIÓN DE INFERENCIA Y EVALUACIÓN (Fase 2) ---

# Umbral de descarte (NONE implícito)
CONFIDENCE_THRESHOLD = 0.75

# Cantidad de fragmentos de texto que BERT procesará en paralelo por cada iteración.
# Un valor de 16 optimiza el uso de la GPU (VRAM) sin riesgo de saturar la memoria (Out of Memory).
BATCH_SIZE_INFERENCE = 16

# Cuántos candidatos buscar por contrato (Recall@K)
TOP_K_CANDIDATES = 3

