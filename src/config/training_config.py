"""
Configuración experimental del modelo.
Permite modificar fácilmente el tamaño del dataset
y los porcentajes de partición.
"""

# número de contratos a utilizar
NUM_CONTRACTS = 100

# proporciones del split
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# --- REPRODUCIBILIDAD ---
# Semilla global para asegurar que la fragmentación y el muestreo sean deterministas
RANDOM_SEED = 42

# split_dataset_stratified
# Cuota mínima de documentos por categoría para garantizar representación de clases minoritarias en todos los splits.
MIN_SAMPLES_PER_CLASS = 10