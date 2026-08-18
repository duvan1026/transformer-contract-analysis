"""
Funciones puras de conteo de longitud de texto (tokens BERT y palabras).
Aisladas para reutilización: hoy las consume analyze_token_length.py, pero
el método de conteo de tokens replica exactamente el usado en
fragmentation_utils.py / quality_filters.py (tokenizer.encode con
add_special_tokens=False), de modo que las cifras sean comparables entre
train/val/test y los fragmentos ya generados en inferencia.
"""


def contar_palabras(texto: str) -> int:
    """
    Conteo de palabras por separación en espacios en blanco.
    Usado como criterio de referencia para detectar fragmentos anómalamente largos.
    """
    if not isinstance(texto, str):
        return 0
    return len(texto.split())


def contar_tokens(texto: str, tokenizer) -> int:
    """
    Conteo de tokens reales del tokenizer del modelo (BERT).
    Mismo método que fragment_single_document: tokenizer.encode(texto, add_special_tokens=False).
    """
    if not isinstance(texto, str):
        return 0
    return len(tokenizer.encode(texto, add_special_tokens=False))