"""
src/utils/rebalancing/io.py

Adaptadores pandas del rebalanceo de spans/fragmentos largos. Traducen entre
el esquema de columnas de train/val/test y el de los fragmentos de
inferencia hacia/desde el núcleo puro de splitting.py (rebalancear_unidad),
que no conoce pandas ni nombres de columna.

Ambas funciones comparten exactamente el mismo algoritmo de división
(splitting.rebalancear_unidad) -- la única diferencia entre ellas es qué
columnas leen y qué columnas escriben.

rebalancear_split_train() SÍ distingue por categoría (ver
REBALANCE_INCLUDE_POSITIVE_CATEGORIES): por defecto, únicamente las filas
NONE se dividen. rebalancear_fragmentos_inferencia() nunca distingue por
categoría, porque en el momento en que corre todavía no existe ninguna
etiqueta -- se calcula después, en 03_calculate_ground_truth.py.
"""

import pandas as pd

from src.utils.rebalancing.splitting import rebalancear_unidad
from src.utils.quality_filters import FragmentQualityFilter
from src.config.text_standards import (
    FRAGMENT_REBALANCE_TARGET_TOKENS,
    FRAGMENT_REBALANCE_MAX_TOKENS,
    REBALANCE_INCLUDE_POSITIVE_CATEGORIES,
)

_METODO_SIN_DIVIDIR = "sin_dividir"


def rebalancear_split_train(df_split: pd.DataFrame, tokenizer, max_tokens, umbral_objetivo) -> tuple[pd.DataFrame, pd.DataFrame]:

    filas_resultado = []
    filas_rechazadas = []

    for unidad_original_id, fila in enumerate(df_split.to_dict(orient="records"), start=1):
        texto = str(fila.get("span_text", ""))
        offset_val = fila.get("offset")
        categoria = str(fila.get("categoria", "")).strip().lower()

        es_positiva = categoria != "none"
        excluir_por_categoria = es_positiva and not REBALANCE_INCLUDE_POSITIVE_CATEGORIES

        if excluir_por_categoria or pd.isna(offset_val) or int(offset_val) < 0 or not texto.strip():
            # Categoría positiva excluida por configuración, o caso
            # defensivo sin offset válido -> se conserva igual, sin dividir.
            fila_out = dict(fila)
            fila_out.update({
                "unidad_original_id": unidad_original_id,
                "subspan_id": 1,
                "metodo_division": _METODO_SIN_DIVIDIR,
            })
            filas_resultado.append(fila_out)
            continue

        piezas = rebalancear_unidad(texto, int(offset_val), tokenizer, max_tokens, umbral_objetivo)
        se_dividio = len(piezas) > 1

        for subspan_id, pieza in enumerate(piezas, start=1):
            if se_dividio:
                es_valido, motivo = FragmentQualityFilter.validate(pieza["texto"])
                if not es_valido:
                    fila_rechazo = dict(fila)
                    fila_rechazo.update({
                        "unidad_original_id": unidad_original_id,
                        "subspan_id": subspan_id,
                        "span_text": pieza["texto"],
                        "offset": pieza["start_offset"],
                        "metodo_division": pieza["metodo_division"],
                        "motivo_descarte": motivo,
                    })
                    filas_rechazadas.append(fila_rechazo)
                    continue

            fila_out = dict(fila)
            fila_out.update({
                "span_text": pieza["texto"],
                "offset": pieza["start_offset"],
                "offset_end": pieza["end_offset"],
                "unidad_original_id": unidad_original_id,
                "subspan_id": subspan_id,
                "metodo_division": pieza["metodo_division"],
                "tokens_estimados": pieza["tokens"],
            })
            filas_resultado.append(fila_out)

    return pd.DataFrame(filas_resultado), pd.DataFrame(filas_rechazadas)


def rebalancear_fragmentos_inferencia(df_fragmentos: pd.DataFrame, tokenizer,  max_tokens, umbral_objetivo) -> tuple[pd.DataFrame, pd.DataFrame]:

    filas_resultado = []
    filas_rechazadas = []

    for fila in df_fragmentos.to_dict(orient="records"):
        texto = str(fila.get("contenido", ""))
        start_offset = fila.get("start_offset")

        if pd.isna(start_offset) or not texto.strip():
            fila_out = dict(fila)
            fila_out.update({
                "subfragmento_id": 1,
                "metodo_division": _METODO_SIN_DIVIDIR,
            })
            filas_resultado.append(fila_out)
            continue

        piezas = rebalancear_unidad(texto, int(start_offset), tokenizer, max_tokens, umbral_objetivo)
        se_dividio = len(piezas) > 1

        for subfragmento_id, pieza in enumerate(piezas, start=1):
            if se_dividio:
                es_valido, motivo = FragmentQualityFilter.validate(pieza["texto"])
                if not es_valido:
                    fila_rechazo = dict(fila)
                    fila_rechazo.update({
                        "subfragmento_id": subfragmento_id,
                        "contenido": pieza["texto"],
                        "start_offset": pieza["start_offset"],
                        "end_offset": pieza["end_offset"],
                        "metodo_division": pieza["metodo_division"],
                        "motivo_descarte": motivo,
                    })
                    filas_rechazadas.append(fila_rechazo)
                    continue

            fila_out = dict(fila)
            fila_out.update({
                "contenido": pieza["texto"],
                "start_offset": pieza["start_offset"],
                "end_offset": pieza["end_offset"],
                "tokens_estimados": pieza["tokens"],
                "subfragmento_id": subfragmento_id,
                "metodo_division": pieza["metodo_division"],
            })
            filas_resultado.append(fila_out)

    return pd.DataFrame(filas_resultado), pd.DataFrame(filas_rechazadas)