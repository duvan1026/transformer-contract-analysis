"""
Carga del ground truth (GROUND_TRUTH_BASELINE / GROUND_TRUTH_BALANCED) y
preparacion de las columnas necesarias para calcular recall por longitud
(spans positivos) y especificidad por longitud (spans NONE), de forma
simetrica: ambas metricas responden "¿el modelo clasifico bien este span,
segun su longitud?", solo cambia que grupo de spans se evalua.

Reutiliza normalize_category (text_utils.py) para 'etiqueta_real' y
'pred_raw', evitando que una inconsistencia de casing entre columnas
cuente como error un acierto real del modelo.
"""

import pandas as pd

from src.utils.text_utils import normalize_category


def _leer_y_normalizar(path, columna_prediccion):
    df = pd.read_csv(path)
    columnas_requeridas = {"etiqueta_real", columna_prediccion, "tokens_estimados"}
    faltantes = columnas_requeridas - set(df.columns)
    if faltantes:
        raise KeyError(
            f"Faltan columnas requeridas en {path}: {faltantes}. "
            f"Verifica el nombre de la columna de prediccion (se uso '{columna_prediccion}')."
        )
    df = df.copy()
    df["etiqueta_real_norm"] = df["etiqueta_real"].apply(normalize_category)
    df["prediccion_norm"] = df[columna_prediccion].apply(normalize_category)
    return df


def cargar_ground_truth_positivos(path, columna_prediccion="pred_raw"):
    """
    Spans reales positivos (etiqueta_real != NONE). correcto = prediccion
    coincide con la clase real. Base para la curva de RECALL por longitud.
    Devuelve columnas: clase, correcto (bool), tokens_estimados.
    """
    try:
        df = _leer_y_normalizar(path, columna_prediccion)
    except FileNotFoundError:
        print(f"  [AVISO] No se encontro {path}.")
        return pd.DataFrame(columns=["clase", "correcto", "tokens_estimados"])

    positivos = df[df["etiqueta_real_norm"] != "none"].copy()
    positivos["correcto"] = positivos["prediccion_norm"] == positivos["etiqueta_real_norm"]
    return positivos[["etiqueta_real_norm", "correcto", "tokens_estimados"]].rename(
        columns={"etiqueta_real_norm": "clase"}
    ).reset_index(drop=True)


def cargar_ground_truth_none(path, columna_prediccion="pred_raw"):
    """
    Spans reales de NONE. correcto = el modelo predijo NONE (verdadero
    negativo). Base para la curva de ESPECIFICIDAD por longitud — el
    equivalente de recall, pero del lado de los falsos positivos.
    Devuelve columnas: clase ('none'), correcto (bool), tokens_estimados.
    """
    try:
        df = _leer_y_normalizar(path, columna_prediccion)
    except FileNotFoundError:
        print(f"  [AVISO] No se encontro {path}.")
        return pd.DataFrame(columns=["clase", "correcto", "tokens_estimados"])

    none_rows = df[df["etiqueta_real_norm"] == "none"].copy()
    none_rows["correcto"] = none_rows["prediccion_norm"] == "none"
    return none_rows[["etiqueta_real_norm", "correcto", "tokens_estimados"]].rename(
        columns={"etiqueta_real_norm": "clase"}
    ).reset_index(drop=True)