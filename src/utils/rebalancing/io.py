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


def rebalancear_split_train(df_split: pd.DataFrame, tokenizer,
                             target_tokens: int = FRAGMENT_REBALANCE_TARGET_TOKENS,
                             max_tokens: int = FRAGMENT_REBALANCE_MAX_TOKENS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aplica el rebalanceo a un split de train/val/test (mismo esquema en los
    tres: id_documento, categoria, span_text, offset, categoria_original).

    Si REBALANCE_INCLUDE_POSITIVE_CATEGORIES es False (valor por defecto),
    las filas cuya categoria no sea "NONE" pasan intactas, sin pasar por
    rebalancear_unidad -- se dejan exactamente como las extrajo
    extract_spans.py desde CUAD. Dividir un span ya etiquetado arriesga que
    el trozo resultante ya no contenga el elemento que justifica esa
    etiqueta (ver text_standards.py para el detalle de este riesgo).

    Añade a cada fila resultante:
        - unidad_original_id: posición (1-indexada) de la fila padre dentro
          de este split. Agrupa las piezas que provienen de un mismo span
          original.
        - subspan_id: 1..k dentro de esa unidad_original_id.
        - metodo_division: nivel de la cascada que produjo la pieza
          ("sin_dividir", "oracion", "clausula_punto_y_coma",
          "sin_dividir_sin_delimitador").
        - tokens_estimados: conteo de tokens de la pieza final.

    Las filas excluidas por categoría reciben metodo_division="sin_dividir"
    igual que una fila que ya cabía en el objetivo, para que la auditoría
    posterior no las distinga de un caso "no necesitaba dividirse" -- el
    resultado final (fila intacta) es el mismo, solo cambia el motivo.

    Retorna
    -------
    (df_resultado, df_rechazadas)
        df_rechazadas puede venir vacío si ninguna pieza nueva fue
        rechazada por el filtro de calidad.
    """
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

        piezas = rebalancear_unidad(texto, int(offset_val), tokenizer, target_tokens, max_tokens)
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
                "unidad_original_id": unidad_original_id,
                "subspan_id": subspan_id,
                "metodo_division": pieza["metodo_division"],
                "tokens_estimados": pieza["tokens"],
            })
            filas_resultado.append(fila_out)

    return pd.DataFrame(filas_resultado), pd.DataFrame(filas_rechazadas)


def rebalancear_fragmentos_inferencia(df_fragmentos: pd.DataFrame, tokenizer,
                                       target_tokens: int = FRAGMENT_REBALANCE_TARGET_TOKENS,
                                       max_tokens: int = FRAGMENT_REBALANCE_MAX_TOKENS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aplica el rebalanceo a batch_fragments.csv (id_documento, parrafo_id,
    fragmento_id, start_offset, end_offset, contenido, tokens_estimados).

    No distingue por categoría -- en este punto del pipeline los fragmentos
    todavía no tienen ninguna etiqueta: se calcula después, por
    solapamiento espacial, en 03_calculate_ground_truth.py.

    `parrafo_id` y `fragmento_id` ya existen y se conservan sin renumerar.
    Se añade únicamente:
        - subfragmento_id: 1..k dentro de ese (parrafo_id, fragmento_id).
        - metodo_division.

    Retorna
    -------
    (df_resultado, df_rechazadas)
    """
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

        piezas = rebalancear_unidad(texto, int(start_offset), tokenizer, target_tokens, max_tokens)
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