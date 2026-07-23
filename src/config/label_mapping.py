"""
Mapping oficial de etiquetas y configuración del modelo.
"""

# Configuración del modelo base
MODEL_NAME = "bert-base-uncased"

# Diccionario para el Baseline (5 clases)
LABEL2ID_BASELINE = {
    "audit rights": 0,
    "governing law": 1,
    "limitation of liability": 2,
    "non-compete": 3,
    "termination for convenience": 4
}
ID2LABEL_BASELINE = {v: k for k, v in LABEL2ID_BASELINE.items()}
NUM_LABELS_BASELINE = len(LABEL2ID_BASELINE)

# Diccionario para Balanced (6 clases)
LABEL2ID_BALANCED = {
    "audit rights": 0,
    "governing law": 1,
    "limitation of liability": 2,
    "non-compete": 3,
    "termination for convenience": 4,
    "none": 5  
}
ID2LABEL_BALANCED = {v: k for k, v in LABEL2ID_BALANCED.items()}
NUM_LABELS_BALANCED = len(LABEL2ID_BALANCED)



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