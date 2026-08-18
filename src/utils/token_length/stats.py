"""
src/utils/token_length_stats.py

Calculo de percentiles/umbrales de longitud (tokens y palabras) agrupados
por clase, y construccion de la tabla comparativa entre fuentes
(train/val/test/inferencia).

Nombres de columna en español, autoexplicativos (sin abreviaturas tipo
"p50" ni IDs de hallazgo internos — esos umbrales viven en
text_standards.py y pueden cambiar). Cada columna de porcentaje va
acompañada de su columna de conteo absoluto equivalente.
"""

import pandas as pd


def calcular_estadisticos(df_fuente, umbral_palabras, zona_riesgo_tokens, max_tokens):
    """
    Agrupa por clase y calcula percentiles/umbrales, en tokens y en palabras.

    umbral_palabras: umbral de palabras de referencia (TOKEN_LENGTH_WORD_THRESHOLD).
    zona_riesgo_tokens: tupla (min, max) del rango de tokens de referencia
                        (TOKEN_LENGTH_RISK_ZONE_TOKENS).
    max_tokens: limite fisico del modelo (MAX_TOKENS), para detectar truncamiento
                en spans de train/val/test que no pasan por fragment_single_document.
    """
    filas = []
    for clase, grupo in df_fuente.groupby("clase"):
        n = len(grupo)

        pct_excede_max = (grupo["n_tokens"] > max_tokens).mean() * 100
        pct_zona_riesgo = (
            (grupo["n_tokens"] >= zona_riesgo_tokens[0]) & (grupo["n_tokens"] <= zona_riesgo_tokens[1])
        ).mean() * 100
        pct_mayor_umbral_palabras = (grupo["n_palabras"] > umbral_palabras).mean() * 100

        fila = {
            "Clase": clase,
            "Cantidad_Fragmentos": n,
            # --- Tokens ---
            "Tokens_Mediana": round(grupo["n_tokens"].quantile(0.50), 1),
            "Tokens_Percentil_75": round(grupo["n_tokens"].quantile(0.75), 1),
            "Tokens_Percentil_90": round(grupo["n_tokens"].quantile(0.90), 1),
            "Tokens_Percentil_95": round(grupo["n_tokens"].quantile(0.95), 1),
            "Tokens_Maximo": int(grupo["n_tokens"].max()),
            "Porcentaje_Excede_Limite_Tokens": round(pct_excede_max, 1),
            "Cantidad_Excede_Limite_Tokens": round(pct_excede_max / 100 * n),
            "Porcentaje_En_Rango_Riesgo_Tokens": round(pct_zona_riesgo, 1),
            "Cantidad_En_Rango_Riesgo_Tokens": round(pct_zona_riesgo / 100 * n),
            # --- Palabras ---
            "Palabras_Mediana": round(grupo["n_palabras"].quantile(0.50), 1),
            "Palabras_Percentil_75": round(grupo["n_palabras"].quantile(0.75), 1),
            "Palabras_Percentil_90": round(grupo["n_palabras"].quantile(0.90), 1),
            "Palabras_Percentil_95": round(grupo["n_palabras"].quantile(0.95), 1),
            "Palabras_Maximo": int(grupo["n_palabras"].max()),
            "Porcentaje_Mayor_Umbral_Palabras": round(pct_mayor_umbral_palabras, 1),
            "Cantidad_Mayor_Umbral_Palabras": round(pct_mayor_umbral_palabras / 100 * n),
        }
        filas.append(fila)
    return pd.DataFrame(filas).sort_values("Clase").reset_index(drop=True)


def construir_resumen_comparativo(stats_por_fuente):
    """
    Combina las tablas de estadisticos de cada fuente en una vista
    clase x fuente, para lectura rapida train vs. inferencia.
    """
    columnas_clave = [
        "Cantidad_Fragmentos", "Tokens_Mediana", "Tokens_Percentil_90", "Tokens_Percentil_95",
        "Porcentaje_En_Rango_Riesgo_Tokens", "Porcentaje_Mayor_Umbral_Palabras",
    ]
    piezas = []
    for fuente, df in stats_por_fuente.items():
        if df.empty:
            continue
        sub = df.set_index("Clase")[columnas_clave].copy()
        sub.columns = [f"{fuente}_{c}" for c in sub.columns]
        piezas.append(sub)
    if not piezas:
        return pd.DataFrame()
    return pd.concat(piezas, axis=1).reset_index()


def construir_tabla_conteos(stats_por_fuente):
    """
    Extrae Clase + Cantidad_Fragmentos de cada fuente y las apila en
    formato largo (fuente, clase, cantidad). Base para graficar el
    volumen bruto de spans/fragmentos por clase y fuente (desbalance
    en cantidad absoluta).
    """
    piezas = []
    for fuente, df in stats_por_fuente.items():
        if df.empty:
            continue
        sub = df[["Clase", "Cantidad_Fragmentos"]].copy()
        sub.insert(0, "fuente", fuente)
        sub = sub.rename(columns={"Clase": "clase", "Cantidad_Fragmentos": "cantidad"})
        piezas.append(sub)
    if not piezas:
        return pd.DataFrame(columns=["fuente", "clase", "cantidad"])
    return pd.concat(piezas, ignore_index=True)


def construir_tabla_composicion(stats_por_fuente, incluir_none=True):
    """
    Convierte Cantidad_Fragmentos en porcentaje relativo dentro de cada
    fuente (cada fuente suma 100%). Responde una pregunta distinta a
    construir_tabla_conteos: no "cuantos hay", sino "que proporcion del
    total de esa fuente representa cada clase" — es la visualizacion
    directa del desajuste de prior entre train/val/test e inferencia.

    incluir_none:
        True  -> composicion completa (incluye NONE). Muestra el
                 desbalance NONE-vs-resto.
        False -> excluye NONE y renormaliza a 100% solo entre las
                 clases positivas, para que el desbalance entre ellas
                 sea visible sin que NONE aplaste la escala.
    """
    piezas = []
    for fuente, df in stats_por_fuente.items():
        if df.empty:
            continue
        sub = df[["Clase", "Cantidad_Fragmentos"]].copy()
        if not incluir_none:
            sub = sub[sub["Clase"] != "none"]
        if sub.empty:
            continue
        total = sub["Cantidad_Fragmentos"].sum()
        if total == 0:
            continue
        sub["porcentaje"] = sub["Cantidad_Fragmentos"] / total * 100
        sub.insert(0, "fuente", fuente)
        sub = sub.rename(columns={"Clase": "clase"})
        piezas.append(sub[["fuente", "clase", "porcentaje"]])
    if not piezas:
        return pd.DataFrame(columns=["fuente", "clase", "porcentaje"])
    return pd.concat(piezas, ignore_index=True)


def construir_tabla_riesgo_truncamiento(stats_por_fuente):
    """
    Extrae Cantidad_Excede_Limite_Tokens por clase, solo de train/val/test
    (inferencia no aplica: fragment_single_document ya garantiza que
    ningun fragmento de inferencia supera MAX_TOKENS por diseño).
    """
    fuentes_aplicables = ["train", "val", "test"]
    piezas = []
    for fuente in fuentes_aplicables:
        df = stats_por_fuente.get(fuente)
        if df is None or df.empty:
            continue
        sub = df[["Clase", "Cantidad_Excede_Limite_Tokens"]].copy()
        sub.insert(0, "fuente", fuente)
        sub = sub.rename(columns={
            "Clase": "clase",
            "Cantidad_Excede_Limite_Tokens": "cantidad_excede",
        })
        piezas.append(sub)
    if not piezas:
        return pd.DataFrame(columns=["fuente", "clase", "cantidad_excede"])
    return pd.concat(piezas, ignore_index=True)


def construir_tabla_riesgo_por_clase_inferencia(stats_por_fuente):
    """
    Extrae, solo de la fuente 'inferencia', el porcentaje de fragmentos de
    cada clase que cae en la zona de riesgo (Porcentaje_En_Rango_Riesgo_Tokens),
    ordenado de mayor a menor. Es el numero mas directamente accionable del
    analisis completo: responde "que clase se beneficiaria mas de una refragmentacion",
    sin tener que leer la tabla completa o inferirlo del ECDF.
    """
    df = stats_por_fuente.get("inferencia")
    if df is None or df.empty:
        return pd.DataFrame(columns=["clase", "porcentaje_riesgo", "cantidad_riesgo", "cantidad_fragmentos"])

    sub = df[[
        "Clase", "Porcentaje_En_Rango_Riesgo_Tokens",
        "Cantidad_En_Rango_Riesgo_Tokens", "Cantidad_Fragmentos",
    ]].copy()
    sub = sub.rename(columns={
        "Clase": "clase",
        "Porcentaje_En_Rango_Riesgo_Tokens": "porcentaje_riesgo",
        "Cantidad_En_Rango_Riesgo_Tokens": "cantidad_riesgo",
        "Cantidad_Fragmentos": "cantidad_fragmentos",
    })
    return sub.sort_values("porcentaje_riesgo", ascending=False).reset_index(drop=True)