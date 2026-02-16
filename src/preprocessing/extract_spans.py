import json
import csv
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DATA = BASE_DIR / "data" / "raw" / "CUAD_v1.json"
OUTPUT_DATA = BASE_DIR / "data" / "processed" / "dataset_spans.csv"


def split_id(full_id: str):
    """Separa id_documento y categoria"""
    return full_id.split("__", 1)

def clean_text(text: str) -> str:
    # Reemplaza saltos de línea múltiples por un espacio
    text = re.sub(r"\n+", " ", text)

    # Colapsa espacios múltiples en uno solo
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_spans():

    rows = []

    with open(RAW_DATA, "r", encoding="utf-8") as f:
        data = json.load(f)

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

                # Caso normal
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
    with open(OUTPUT_DATA, "w", newline="", encoding="utf-8") as f:
        # writer = csv.writer(f)

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

if __name__ == "__main__":
    extract_spans()