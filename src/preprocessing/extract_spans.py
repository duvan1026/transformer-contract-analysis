import sys
from pathlib import Path
import json
import csv

# Añade la raíz del proyecto al path de Python para poder importar módulos desde "src"
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config.paths import CUAD_JSON, DATASET_SPANS_CSV
from src.utils.text_utils import split_id, clean_text

def extract_spans():
    """
    Extrae spans desde el dataset CUAD en formato JSON y genera un CSV en formato largo,
    donde cada fila representa un span individual.
    """

    rows = []

    # Cargar JSON
    with open(CUAD_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Recorrer estructura del dataset
    for document in data["data"]:
        for paragraph in document["paragraphs"]:
            for qa in paragraph["qas"]:

                id_documento, categoria = split_id(qa["id"])
                question = qa["question"]
                is_impossible = qa["is_impossible"]
                answers = qa["answers"]

                # Caso sin respuestas
                if is_impossible or len(answers) == 0:
                    rows.append([
                        id_documento,
                        categoria,
                        question,
                        is_impossible,
                        "",
                        -1
                    ])

                # Caso con múltiples spans
                else:
                    for answer in answers:
                        rows.append([
                            id_documento,
                            categoria,
                            question,
                            is_impossible,
                            clean_text(answer["text"]),
                            answer["answer_start"]
                        ])

    # Guardar CSV
    with open(DATASET_SPANS_CSV, "w", newline="", encoding="utf-8") as f:
        
        # Crea un escritor CSV que usa comas como separadores y encierra automáticamente
        # cada campo entre comillas para manejar correctamente spans con comas o caracteres especiales
        writer = csv.writer(
            f,
            delimiter=",",
            quoting=csv.QUOTE_ALL
        )

        writer.writerow([
            "id_documento",
            "categoria",
            "question",
            "is_impossible",
            "span_text",
            "offset"
        ])

        writer.writerows(rows)

def run():
    extract_spans()


if __name__ == "__main__":
    run()