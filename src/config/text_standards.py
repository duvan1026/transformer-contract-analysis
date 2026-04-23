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

# Configuración del modelo base
MODEL_NAME = "bert-base-uncased"

# Cantidad de documentos para procesar en un lote de inferencia
BATCH_SIZE_DOCS = 100
EXPERIMENT_NAME = "v2_inferencia_100_docs"

# --- FLUJO DE DOCUMENTOS (Registro Maestro) ---
DOC_STATES = {
    "AVAILABLE": "Disponible",
    "IN_PROGRESS": "En_proceso",
    "PROCESSED": "Utilizado",
    "NONE": "Ninguno"
}