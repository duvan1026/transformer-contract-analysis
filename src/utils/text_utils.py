"""
text_utils.py

Funciones auxiliares para el procesamiento y limpieza de texto
utilizadas en el pipeline de preprocessing del proyecto.
"""

import re
import pandas as pd

def split_id(full_id: str) -> tuple[str, str]:
    """
    Separa el identificador compuesto del dataset CUAD en:

    - id del documento
    - categoría de la cláusula

    Ejemplo:
        "DOC123__Governing Law"
        → ("DOC123", "Governing Law")
    """

    if "__" not in full_id:
        raise ValueError(f"Formato de ID inválido: {full_id}")

    return full_id.split("__", 1)


def clean_text(text: str) -> str:
    """
    Normaliza texto eliminando:

    - saltos de línea múltiples
    - espacios redundantes

    Mantiene el contenido semántico intacto.
    """

    if not isinstance(text, str):
        return ""

    # Reemplaza múltiples saltos de línea por espacio
    text = re.sub(r"\n+", " ", text)

    # Colapsa espacios múltiples
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def is_empty_text(text: str) -> bool:
    """
    Verifica si un texto está vacío o contiene solo espacios.
    """

    return not text or text.strip() == ""

def normalize_category(text):
    if pd.isna(text):
        return ""
    return str(text).strip().lower()