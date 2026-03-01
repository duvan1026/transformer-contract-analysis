import pandas as pd
from src.utils.text_utils import normalize_category


def load_category_mapping(mapping_file):
    """
    Carga el archivo Excel de homologación y construye un diccionario
    de mapeo entre categorías originales y categorías finales.

    - Lee el archivo de homologación
    - Filtra las filas marcadas como "Usar = Sí"
    - Normaliza los nombres de categorías
    - Retorna un diccionario {categoria_original: categoria_final}

    Parámetros
    ----------
    mapping_file : Path o str
        Ruta al archivo Excel.

    Retorna
    -------
    dict
        Diccionario de mapeo de categorías.
    """

    df = pd.read_excel(mapping_file)

    df = df[df["Usar"] == "Sí"].copy()

    df["Category"] = df["Category"].apply(normalize_category)
    df["Final Name Category"] = df["Final Name Category"].apply(normalize_category)

    return dict(
        zip(
            df["Category"],
            df["Final Name Category"]
        )
    )


def normalize_categories_column(df, column):
    """
    Normaliza los valores de una columna de categorías en un DataFrame
    para garantizar consistencia de formato.

    Parámetros
    ----------
    df : pandas.DataFrame
        DataFrame de entrada.

    column : str
        Nombre de la columna a normalizar.

    Retorna
    -------
    pandas.DataFrame
        DataFrame con la columna normalizada.
    """

    df[column] = df[column].apply(normalize_category)

    return df


def apply_category_mapping(df, mapping, column="categoria"):
    """
    Filtra las categorías válidas y aplica el mapeo a la taxonomía final.
    Conserva la categoría original en una nueva columna.

    Parámetros
    ----------
    df : pandas.DataFrame
        DataFrame con las categorías originales.

    mapping : dict
        Diccionario de homologación.

    column : str, opcional
        Nombre de la columna de categorías.

    Retorna
    -------
    pandas.DataFrame
        DataFrame con categorías mapeadas.
    """

    df = df.copy()

    df = df[df[column].isin(mapping.keys())]

    df["categoria_original"] = df[column]

    df[column] = df[column].map(mapping)

    return df