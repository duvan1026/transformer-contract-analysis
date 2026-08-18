"""
src/utils/recall_by_length/stats.py

Calculo de recall por rango de longitud y determinacion automatica de la
zona de menor acierto, en vez de asumir un rango fijo copiado de un
analisis anterior.
"""

import pandas as pd


def calcular_recall_por_bin(df_positivos, bins):
    """
    Agrupa los spans reales positivos por rango de tokens_estimados y
    calcula el recall (promedio de 'correcto') y el tamaño de muestra
    de cada bin: pd.cut con los cortes indicados, groupby + mean/count.

    include_lowest=True asegura que un fragmento con tokens_estimados
    exactamente igual al limite inferior del primer bin (0) no quede
    excluido silenciosamente (riesgo detectado al revisar el calculo
    original, que usaba pd.cut sin este parametro).
    """
    if df_positivos.empty:
        return pd.DataFrame(columns=["bin", "bin_min", "bin_max", "recall", "n"])

    df = df_positivos.copy()
    df["bin"] = pd.cut(df["tokens_estimados"], bins=bins, include_lowest=True)

    fuera_de_rango = df["bin"].isna().sum()
    if fuera_de_rango > 0:
        print(f"  [AVISO] {fuera_de_rango} spans positivos quedan fuera de los bins "
              f"definidos ({bins[0]}-{bins[-1]} tokens) y se excluyen del calculo.")

    resumen = (
        df.dropna(subset=["bin"])
        .groupby("bin", observed=True)["correcto"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "recall", "count": "n"})
    )
    resumen["recall"] = (resumen["recall"] * 100).round(1)
    resumen["bin_min"] = resumen["bin"].apply(lambda b: int(b.left))
    resumen["bin_max"] = resumen["bin"].apply(lambda b: int(b.right))
    resumen["bin_label"] = resumen.apply(lambda r: f"{r['bin_min']}-{r['bin_max']}", axis=1)

    return resumen[["bin_label", "bin_min", "bin_max", "recall", "n"]]


def determinar_zona_riesgo(df_recall, n_minimo=15):
    """
    Determina la zona de riesgo como el bin con menor recall, entre los
    bins que tienen al menos n_minimo muestras (para no elegir un bin
    con recall extremo solo por tener pocos datos, poco confiable).
    Si ningun bin alcanza n_minimo, se usa el conjunto completo con aviso.

    Devuelve un dict: {bin_min, bin_max, recall, n} o None si no hay datos.
    """
    if df_recall.empty:
        return None

    candidatos = df_recall[df_recall["n"] >= n_minimo]
    if candidatos.empty:
        print(f"  [AVISO] Ningun bin alcanza el minimo de {n_minimo} muestras. "
              f"Se determina la zona de riesgo sobre todos los bins disponibles, "
              f"con menor confiabilidad estadistica.")
        candidatos = df_recall

    fila = candidatos.loc[candidatos["recall"].idxmin()]
    return {
        "bin_min": int(fila["bin_min"]),
        "bin_max": int(fila["bin_max"]),
        "recall": float(fila["recall"]),
        "n": int(fila["n"]),
    }


def ponderar_metrica_por_distribucion(df_metrica_bin, tokens_series, bins):
    """
    Aplica una curva de metrica por bin (recall o especificidad, ya
    calculada con calcular_recall_por_bin) sobre una distribucion real de
    longitudes, y devuelve el promedio ponderado: cuanto acertaria el
    modelo, en promedio, dado como se distribuyen esas longitudes.

    Se usa dos veces con la misma curva: una vez con la distribucion real
    de inferencia (escenario actual) y otra con la de train (escenario
    objetivo, si la longitud de inferencia se pareciera a la de train).

    Devuelve (valor_ponderado_pct, n_total) o (None, 0) si no hay datos.
    """
    if df_metrica_bin.empty or tokens_series is None or len(tokens_series) == 0:
        return None, 0

    dist = pd.DataFrame({"tokens": tokens_series})
    dist["bin"] = pd.cut(dist["tokens"], bins=bins, include_lowest=True)
    resumen = dist.dropna(subset=["bin"]).groupby("bin", observed=True).size().reset_index(name="n")
    if resumen.empty:
        return None, 0

    resumen["bin_min"] = resumen["bin"].apply(lambda b: int(b.left))
    resumen["bin_max"] = resumen["bin"].apply(lambda b: int(b.right))
    total_n = int(resumen["n"].sum())
    resumen["proporcion"] = resumen["n"] / total_n

    merged = resumen.merge(df_metrica_bin[["bin_min", "bin_max", "recall"]], on=["bin_min", "bin_max"], how="left")
    merged["recall"] = merged["recall"].fillna(0)  # bin sin dato de la curva -> supuesto conservador (0% acierto)

    valor = (merged["proporcion"] * merged["recall"]).sum()
    return round(valor, 1), total_n