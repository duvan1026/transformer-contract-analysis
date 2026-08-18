"""
Carga de las distintas fuentes (train/val/test/inferencia), normalización de
etiquetas y cálculo de longitud (tokens/palabras) por fila.

Reutiliza:
- normalize_category (text_utils.py) para la normalización de etiquetas,
  en vez de mantener una versión propia — misma regla que ya usa
  category_utils.py en el resto del proyecto.
- contar_tokens / contar_palabras (tokenization_utils.py).
"""

import pandas as pd

from src.utils.text_utils import normalize_category
from src.utils.token_length.tokenization import contar_tokens, contar_palabras


def _registrar_no_reconocidas(df_no_reconocidas, nombre_fuente, columna_etiqueta,
                               columna_id, registro_no_reconocidas):
    """
    Construye el resumen de etiquetas no reconocidas para una fuente dada
    y lo agrega a la lista acumuladora (no se descarta nada en silencio).
    """
    if len(df_no_reconocidas) == 0:
        return

    resumen = (
        df_no_reconocidas
        .assign(valor_original=df_no_reconocidas[columna_etiqueta])
        .groupby(["valor_original", "clase_normalizada"], dropna=False)
        .size()
        .reset_index(name="conteo")
    )
    resumen.insert(0, "fuente", nombre_fuente)

    if columna_id and columna_id in df_no_reconocidas.columns:
        ejemplos = (
            df_no_reconocidas.groupby(columna_etiqueta)[columna_id]
            .first()
            .rename("id_documento_ejemplo")
            .reset_index()
        )
        resumen = resumen.merge(
            ejemplos, left_on="valor_original", right_on=columna_etiqueta, how="left"
        ).drop(columns=[columna_etiqueta])

    registro_no_reconocidas.append(resumen)
    print(f"  [AVISO] {nombre_fuente}: {len(df_no_reconocidas)} filas con etiqueta "
          f"no reconocida, excluidas de las estadisticas (ver auditoria).")


def cargar_split(path, nombre_fuente, columna_texto, columna_etiqueta,
                  etiquetas_validas, tokenizer, registro_no_reconocidas):
    """
    Carga un CSV de train/val/test, normaliza etiquetas (normalize_category),
    calcula longitud en palabras y tokens, y separa filas con etiquetas no
    reconocidas en vez de descartarlas en silencio.
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"  [AVISO] No se encontro {path} - se omite la fuente '{nombre_fuente}'.")
        return pd.DataFrame(columns=["fuente", "clase", "n_palabras", "n_tokens"])

    df = df.copy()
    df["clase_normalizada"] = df[columna_etiqueta].apply(normalize_category)

    mask_valida = df["clase_normalizada"].isin(etiquetas_validas)
    df_no_reconocidas = df.loc[~mask_valida]
    _registrar_no_reconocidas(
        df_no_reconocidas, nombre_fuente, columna_etiqueta,
        "id_documento" if "id_documento" in df.columns else None,
        registro_no_reconocidas
    )

    df_validas = df.loc[mask_valida].copy()
    df_validas["n_palabras"] = df_validas[columna_texto].apply(contar_palabras)
    df_validas["n_tokens"] = df_validas[columna_texto].apply(
        lambda t: contar_tokens(t, tokenizer)
    )

    return pd.DataFrame({
        "fuente": nombre_fuente,
        "clase": df_validas["clase_normalizada"],
        "n_palabras": df_validas["n_palabras"],
        "n_tokens": df_validas["n_tokens"],
    })


def cargar_inferencia_ground_truth(path, etiquetas_validas, registro_no_reconocidas):
    """
    Carga el ground truth de inferencia (GROUND_TRUTH_BASELINE / GROUND_TRUTH_BALANCED).
    A diferencia de train/val/test, reutiliza la columna 'tokens_estimados' ya
    calculada por fragmentation_utils.py (mismo tokenizer, mismo metodo) en vez
    de retokenizar 'contenido'.
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"  [AVISO] No se encontro {path} - se omite la fuente 'inferencia'.")
        return pd.DataFrame(columns=["fuente", "clase", "n_palabras", "n_tokens"])

    df = df.copy()
    df["clase_normalizada"] = df["etiqueta_real"].apply(normalize_category)

    mask_valida = df["clase_normalizada"].isin(etiquetas_validas)
    df_no_reconocidas = df.loc[~mask_valida]
    _registrar_no_reconocidas(
        df_no_reconocidas, "inferencia", "etiqueta_real", "id_documento",
        registro_no_reconocidas
    )

    df_validas = df.loc[mask_valida].copy()
    df_validas["n_palabras"] = df_validas["contenido"].apply(contar_palabras)
    df_validas["n_tokens"] = df_validas["tokens_estimados"]  # reutilizado, no retokenizado

    return pd.DataFrame({
        "fuente": "inferencia",
        "clase": df_validas["clase_normalizada"],
        "n_palabras": df_validas["n_palabras"],
        "n_tokens": df_validas["n_tokens"],
    })