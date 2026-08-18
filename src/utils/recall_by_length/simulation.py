"""
src/utils/recall_by_length/simulation.py

Estima el impacto de que la longitud de los fragmentos de inferencia se
pareciera a la de entrenamiento: compara, por clase, el recall/especificidad
ponderado bajo la distribucion real actual vs. bajo la distribucion de train
(escenario objetivo), y deriva macro-recall y accuracy globales estimados.

Limitacion explicita: NO estima macro-F1. F1 requiere precision por clase,
que depende de como se reparten los falsos positivos entre las clases
positivas — informacion que no esta disponible en el ground truth agregado
que se usa aqui. Esta simulacion es un techo estimado bajo el supuesto de
que la relacion longitud-acierto observada hoy se mantiene igual con
fragmentos nuevos mas cortos; no reemplaza una validacion real con F02/F03
implementado y reentrenado.
"""

import pandas as pd

from src.utils.recall_by_length.stats import ponderar_metrica_por_distribucion


def construir_tabla_impacto(df_recall_positivos, df_especificidad_none, distribuciones, bins):
    """
    distribuciones: dict {clase: {"actual": Series[tokens], "objetivo": Series[tokens] o None}}
    Incluye las 5 clases positivas y 'none'.

    Devuelve (tabla, resumen_global):
      tabla: DataFrame con clase, metrica_actual_pct, metrica_objetivo_pct, delta_pp, n_actual
      resumen_global: dict con macro_recall_actual/objetivo y accuracy_actual/objetivo
    """
    filas = []
    for clase, dist in distribuciones.items():
        curva = df_especificidad_none if clase == "none" else df_recall_positivos

        actual_pct, n_actual = ponderar_metrica_por_distribucion(curva, dist.get("actual"), bins)
        objetivo_pct, _ = ponderar_metrica_por_distribucion(curva, dist.get("objetivo"), bins)

        delta = None
        if actual_pct is not None and objetivo_pct is not None:
            delta = round(objetivo_pct - actual_pct, 1)

        filas.append({
            "clase": clase,
            "metrica_actual_pct": actual_pct,
            "metrica_objetivo_pct": objetivo_pct,
            "delta_pp": delta,
            "n_actual": n_actual,
        })

    tabla = pd.DataFrame(filas)

    positivas = tabla[(tabla["clase"] != "none") & tabla["metrica_actual_pct"].notna()]
    macro_recall_actual = round(positivas["metrica_actual_pct"].mean(), 1) if not positivas.empty else None
    positivas_obj = positivas[positivas["metrica_objetivo_pct"].notna()]
    macro_recall_objetivo = round(positivas_obj["metrica_objetivo_pct"].mean(), 1) if not positivas_obj.empty else None

    tabla_completa = tabla.dropna(subset=["metrica_actual_pct"])
    total_n = tabla_completa["n_actual"].sum()
    accuracy_actual = accuracy_objetivo = None
    if total_n > 0:
        accuracy_actual = round(
            (tabla_completa["metrica_actual_pct"] / 100 * tabla_completa["n_actual"]).sum() / total_n * 100, 1
        )
        tabla_obj = tabla_completa.dropna(subset=["metrica_objetivo_pct"])
        if len(tabla_obj) == len(tabla_completa):
            accuracy_objetivo = round(
                (tabla_obj["metrica_objetivo_pct"] / 100 * tabla_obj["n_actual"]).sum() / total_n * 100, 1
            )

    resumen_global = {
        "macro_recall_actual": macro_recall_actual,
        "macro_recall_objetivo": macro_recall_objetivo,
        "accuracy_actual": accuracy_actual,
        "accuracy_objetivo": accuracy_objetivo,
    }
    return tabla, resumen_global