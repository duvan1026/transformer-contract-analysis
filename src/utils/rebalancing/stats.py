"""
src/utils/rebalancing/stats.py

Construye el resumen textual del rebalanceo aplicado a un split de
train/val/test o a los fragmentos de inferencia, para impresión por
consola y persistencia en REBALANCE_REPORT_TRAIN / REBALANCE_REPORT_INFERENCE_*.

Asimetría deliberada entre ambas funciones:
- Train SÍ tiene columna 'categoria' -> desglose por clase.
- Inferencia NO tiene ninguna etiqueta en este punto del pipeline (la
  predicción del modelo se calcula después, en 02_run_inference.py) -> el
  desglose posible es por método de división y por documento, no por clase.
  El cruce por clase de los fragmentos rebalanceados solo podrá hacerse más
  adelante contra ground_truth_results.csv, fuera del alcance de este paso.
"""

import pandas as pd

_METODOS_ORDEN = ["oracion", "clausula_punto_y_coma", "clausula_coma", "sin_dividir_sin_delimitador"]


def _linea_metodos(serie_metodo_division: pd.Series) -> str:
    conteo = serie_metodo_division.value_counts()
    partes = [f"{m}={int(conteo.get(m, 0))}" for m in _METODOS_ORDEN]
    return " | ".join(partes)


def resumir_train(df_resultado: pd.DataFrame, df_rechazadas: pd.DataFrame, umbral: int, nombre_split: str) -> str:
    """
    Resumen con desglose por categoría para un split de train/val/test ya
    rebalanceado. `df_resultado` debe incluir 'categoria', 'unidad_original_id'
    y 'metodo_division' (columnas que produce rebalancear_split_train()).
    """
    lineas = []
    lineas.append(f"REBALANCEO DE SPANS LARGOS -- {nombre_split}")
    lineas.append("=" * 60)
    lineas.append(f"Umbral aplicado: {umbral} tokens")
    lineas.append("")

    n_unidades = df_resultado["unidad_original_id"].nunique()
    n_filas = len(df_resultado)
    mask_dividido = df_resultado["metodo_division"] != "sin_dividir"
    n_unidades_divididas = df_resultado.loc[mask_dividido, "unidad_original_id"].nunique()
    n_rechazadas = len(df_rechazadas)

    pct_divididas = (n_unidades_divididas / n_unidades * 100) if n_unidades else 0.0

    lineas.append(f"Unidades originales (spans): {n_unidades}")
    lineas.append(f"Filas resultado (post-rebalanceo): {n_filas}")
    lineas.append(f"Unidades que requirieron división: {n_unidades_divididas} ({pct_divididas:.1f}%)")
    lineas.append(f"Piezas rechazadas por FragmentQualityFilter: {n_rechazadas}")
    lineas.append("")
    lineas.append("Desglose por categoría")
    lineas.append("-" * 40)

    for categoria, grupo in df_resultado.groupby("categoria", dropna=False):
        n_unid_cat = grupo["unidad_original_id"].nunique()
        n_piezas_cat = len(grupo)
        mask_div_cat = grupo["metodo_division"] != "sin_dividir"
        n_unid_div_cat = grupo.loc[mask_div_cat, "unidad_original_id"].nunique()
        pct_cat = (n_unid_div_cat / n_unid_cat * 100) if n_unid_cat else 0.0

        n_rech_cat = 0
        if not df_rechazadas.empty and "categoria" in df_rechazadas.columns:
            n_rech_cat = int((df_rechazadas["categoria"] == categoria).sum())

        lineas.append(f"{categoria}")
        lineas.append(
            f"  Unidades originales: {n_unid_cat} | Divididas: {n_unid_div_cat} ({pct_cat:.1f}%) "
            f"| Piezas resultantes: {n_piezas_cat} | Rechazadas: {n_rech_cat}"
        )
        if n_unid_div_cat > 0:
            lineas.append(f"  Método: {_linea_metodos(grupo.loc[mask_div_cat, 'metodo_division'])}")
        lineas.append("")

    return "\n".join(lineas)


def resumir_inferencia(df_resultado: pd.DataFrame, df_rechazadas: pd.DataFrame, umbral: int,
                        nombre_experimento: str, top_n_documentos: int = 10) -> str:
    """
    Resumen para fragmentos de inferencia ya rebalanceados. Sin desglose por
    clase (ver docstring del módulo). Desglose por método de división y por
    documento (los que más fragmentos nuevos generaron, para priorizar
    revisión manual si aplica).
    """
    lineas = []
    lineas.append(f"REBALANCEO DE FRAGMENTOS -- inferencia ({nombre_experimento})")
    lineas.append("=" * 60)
    lineas.append(f"Umbral aplicado: {umbral} tokens")
    lineas.append("")
    lineas.append(
        "Nota: no hay desglose por clase en esta fase. La predicción del "
        "modelo (pred_raw) todavía no existe -- se calcula en "
        "02_run_inference.py, que corre después de este paso. El cruce por "
        "clase de los fragmentos rebalanceados puede obtenerse más adelante "
        "contra ground_truth_results.csv."
    )
    lineas.append("")

    n_original_ids = df_resultado.groupby(["id_documento", "parrafo_id", "fragmento_id"]).ngroups
    n_filas = len(df_resultado)
    mask_dividido = df_resultado["metodo_division"] != "sin_dividir"
    n_unidades_divididas = df_resultado.loc[mask_dividido].groupby(
        ["id_documento", "parrafo_id", "fragmento_id"]
    ).ngroups
    n_rechazadas = len(df_rechazadas)
    pct_divididas = (n_unidades_divididas / n_original_ids * 100) if n_original_ids else 0.0

    lineas.append(f"Fragmentos originales: {n_original_ids}")
    lineas.append(f"Filas resultado (post-rebalanceo): {n_filas}")
    lineas.append(f"Fragmentos que requirieron división: {n_unidades_divididas} ({pct_divididas:.1f}%)")
    lineas.append(f"Piezas rechazadas por FragmentQualityFilter: {n_rechazadas}")
    lineas.append("")

    lineas.append("Desglose por método de división")
    lineas.append("-" * 40)
    lineas.append(_linea_metodos(df_resultado.loc[mask_dividido, "metodo_division"]))
    lineas.append("")

    if n_unidades_divididas > 0:
        lineas.append(f"Documentos con más fragmentos nuevos generados (top {top_n_documentos})")
        lineas.append("-" * 40)
        # Filas totales por documento menos fragmentos originales (parrafo_id,fragmento_id) únicos de ese documento
        filas_por_doc = df_resultado.groupby("id_documento").size()
        # originales_por_doc = df_resultado.groupby("id_documento").apply(
        #     lambda g: g.groupby(["parrafo_id", "fragmento_id"]).ngroups, include_groups=False
        # )
        originales_por_doc = df_resultado.groupby("id_documento").apply(
            lambda g, **kwargs: g.groupby(["parrafo_id", "fragmento_id"]).ngroups
        )
        extra_por_doc = (filas_por_doc - originales_por_doc).sort_values(ascending=False)
        for doc_id, extra in extra_por_doc.head(top_n_documentos).items():
            if extra > 0:
                lineas.append(f"  {doc_id}: +{int(extra)} fragmentos")
        lineas.append("")

    return "\n".join(lineas)