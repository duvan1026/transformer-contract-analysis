"""
Mapping oficial de etiquetas y configuración del modelo.
"""

# Configuración del modelo base
MODEL_NAME = "bert-base-uncased"

LABEL2ID = {
    "audit rights": 0,
    "governing law": 1,
    "limitation of liability": 2,
    "non-compete": 3,
    "termination for convenience": 4,
    "none": 5  
}

ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# Calculamos NUM_LABELS dinámicamente basándonos en el mapping
NUM_LABELS = len(LABEL2ID)