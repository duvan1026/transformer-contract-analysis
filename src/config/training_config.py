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

# semilla para reproducibilidad
RANDOM_SEED = 42