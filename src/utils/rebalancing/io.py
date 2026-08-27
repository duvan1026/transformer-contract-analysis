"""
src/utils/rebalancing/io.py

Adaptadores pandas del rebalanceo de spans/fragmentos largos. Traducen entre
el esquema de columnas de train/val/test y el de los fragmentos de
inferencia hacia/desde el núcleo puro de splitting.py (rebalancear_unidad),
que no conoce pandas ni nombres de columna.

Ambas funciones comparten exactamente el mismo algoritmo de división
(splitting.rebalancear_unidad) -- la única diferencia entre ellas es qué
columnas leen y qué columnas escriben, cumpliendo la instrucción de que la
misma estrategia se use tanto en entrenamiento como en inferencia.

Todas las filas se procesan por igual, sean positivas o NONE (no se filtra
por columna 'categoria' en ningún punto de este módulo).
"""

import pandas as pd

from src.utils.rebalancing.splitting import rebalancear_unidad
from src.utils.quality_filters import FragmentQualityFilter

_METODO_SIN_DIVIDIR = "sin_dividir"


def rebalancear_split_train(df_split: pd.DataFrame, tokenizer, umbral: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aplica el rebalanceo a un split de train/val/test (mismo esquema en los
    tres: id_documento, categoria, span_text, offset, categoria_original).
    No distingue entre clases positivas y NONE -- todas las filas que
    excedan `umbral` se dividen por igual.

    Añade a cada fila resultante:
        - unidad_original_id: posición (1-indexada) de la fila padre dentro
          de este split. Agrupa las piezas que provienen de un mismo span
          original.
        - subspan_id: 1..k dentro de esa unidad_original_id.
        - metodo_division: nivel de la cascada que produjo la pieza
          ("sin_dividir", "oracion", "clausula_punto_y_coma",
          "clausula_coma", "sin_dividir_sin_delimitador").
        - tokens_estimados: conteo de tokens de la pieza final (para
          auditoría, sin necesidad de retokenizar después).

    Las piezas nuevas resultantes de una división pasan por
    FragmentQualityFilter antes de incluirse; las que ya existían y no
    excedían el umbral NO se re-auditan (se conservan tal cual, igual que
    hoy, sin cambiar comportamiento no solicitado sobre datos que ya
    funcionaban).

    Retorna
    -------
    (df_resultado, df_rechazadas)
        df_rechazadas puede venir vacío (sin filas) si ninguna pieza nueva
        fue rechazada por el filtro de calidad.
    """
    filas_resultado = []
    filas_rechazadas = []

    for unidad_original_id, fila in enumerate(df_split.to_dict(orient="records"), start=1):
        texto = str(fila.get("span_text", ""))
        offset_val = fila.get("offset")

        if pd.isna(offset_val) or int(offset_val) < 0 or not texto.strip():
            # Sin offset válido (caso defensivo -- apply_category_mapping.py
            # ya filtra is_impossible antes de llegar aquí) -> se conserva igual.
            fila_out = dict(fila)
            fila_out.update({
                "unidad_original_id": unidad_original_id,
                "subspan_id": 1,
                "metodo_division": _METODO_SIN_DIVIDIR,
            })
            filas_resultado.append(fila_out)
            continue

        piezas = rebalancear_unidad(texto, int(offset_val), tokenizer, umbral)
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


def rebalancear_fragmentos_inferencia(df_fragmentos: pd.DataFrame, tokenizer, umbral: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aplica el rebalanceo a batch_fragments.csv (id_documento, parrafo_id,
    fragmento_id, start_offset, end_offset, contenido, tokens_estimados).

    A diferencia de train, aquí `parrafo_id` y `fragmento_id` YA EXISTEN y
    se conservan sin renumerar -- identifican la unidad original de la que
    proviene cada pieza. Se añade únicamente:
        - subfragmento_id: 1..k dentro de ese (parrafo_id, fragmento_id).
        - metodo_division.

    Misma lógica de auditoría de calidad que rebalancear_split_train: solo
    se valida con FragmentQualityFilter la pieza que resulta de una
    división real.

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

        piezas = rebalancear_unidad(texto, int(start_offset), tokenizer, umbral)
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