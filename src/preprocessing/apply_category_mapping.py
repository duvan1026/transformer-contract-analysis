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

from src.utils.text_utils import normalize_category


def load_mapping():
    df = pd.read_excel(CATEGORY_MAPPING_FILE)

    df = df[df["Usar"] == "Sí"]

    df["Category"] = df["Category"].apply(normalize_category)
    df["Final Name Category"] = df["Final Name Category"].apply(normalize_category)


    # Crear diccionario
    mapping = dict(
        zip(
            df["Category"],
            df["Final Name Category"]
        )
    )

    return mapping


def apply_mapping():

    print("Loading dataset_spans.csv...")
    spans = pd.read_csv(DATASET_SPANS_CSV)

    spans["categoria"] = spans["categoria"].apply(normalize_category)

    mapping = load_mapping()

    spans = spans[
        spans["categoria"].isin(mapping.keys())
    ]

    spans["categoria_original"] = spans["categoria"]

    spans["categoria"] = spans["categoria"].map(mapping)

    spans.to_csv(
        DATASET_TAXONOMY_CSV,
        index=False
    )


def run():
    apply_mapping()


if __name__ == "__main__":
    run()