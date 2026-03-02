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
    2. Filtrar spans válidos (is_impossible = False)
    3. Normalizar nombres de categorías
    4. Cargar archivo de homologación
    5. Filtrar y mapear a las categorías finales
    6. Guardar el dataset listo para entrenamiento
    """

    # 1. Cargar dataset original
    spans = pd.read_csv(DATASET_SPANS_CSV)

    # 2. Filtrar solo spans válidos (que sí tienen respuesta)
    if "is_impossible" in spans.columns:
        spans = spans[spans["is_impossible"] == False].copy()

    # 3. Normalizar categorías para evitar inconsistencias de formato
    spans = normalize_categories_column(
        spans,
        "categoria"
    )

    # 4. Cargar mapping definido en el archivo de taxonomía
    mapping = load_category_mapping(
        CATEGORY_MAPPING_FILE
    )

    # 5. Filtrar categorías seleccionadas y aplicar homologación
    spans = apply_category_mapping(
        spans,
        mapping
    )

    # 6. Eliminar columnas que NO se usarán en el entrenamiento
    spans = spans.drop(
        columns=[
            "question",
            "is_impossible"
        ],
        errors="ignore"
    )

    # 7. Guardar dataset final
    spans.to_csv(
        DATASET_TAXONOMY_CSV,
        index=False
    )

    print(spans.shape)
    print(spans.isnull().sum())
    print(spans.duplicated().sum())
    print(spans["categoria"].value_counts())


if __name__ == "__main__":
    run()