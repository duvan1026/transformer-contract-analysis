import sys
from pathlib import Path
import pandas as pd

# añadir root del proyecto
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config.paths import (
    DATASET_SPANS_CSV,
    CATEGORY_MAPPING_FILE,
    DATASET_TAXONOMY_CSV
)

from src.utils.category_utils import (
    load_category_mapping,
    normalize_categories_column,
    apply_category_mapping
)

def run():
    """
    Aplica la homologación de categorías definida en la taxonomía
    al dataset de spans extraído desde CUAD.

    Flujo:
    1. Cargar dataset original de spans
    2. Normalizar nombres de categorías
    3. Cargar archivo de homologación
    4. Filtrar y mapear a las categorías finales
    5. Guardar el dataset listo para entrenamiento
    """

    # 1. Cargar dataset original
    spans = pd.read_csv(DATASET_SPANS_CSV)

    # 2. Normalizar categorías para evitar inconsistencias de formato
    spans = normalize_categories_column(
        spans,
        "categoria"
    )

    # 3. Cargar mapping definido en el archivo de taxonomía
    mapping = load_category_mapping(
        CATEGORY_MAPPING_FILE
    )

    # 4. Filtrar categorías seleccionadas y aplicar homologación
    spans = apply_category_mapping(
        spans,
        mapping
    )

    # 5. Guardar dataset final
    spans.to_csv(
        DATASET_TAXONOMY_CSV,
        index=False
    )


if __name__ == "__main__":
    run()