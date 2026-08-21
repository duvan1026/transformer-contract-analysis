"""
src/utils/metrics_utils.py

Módulo analítico central. Contiene todas las reglas de negocio,
simulación de umbrales y cálculo de métricas de desempeño del modelo.
"""

import pandas as pd
from sklearn.metrics import classification_report

def apply_dynamic_threshold(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Regla de Negocio para el modelo Baseline.
    Simula qué pasaría si aplicamos un umbral de confianza estricto.
    Si la confianza no supera el umbral, degrada la predicción a NONE.
    """
    df_sim = df.copy()

    # Degradación condicional (numpy.where logic)
    df_sim["pred_final"] = df_sim.apply(
        lambda x: x["pred_raw"] if x["confidence_raw"] >= threshold else "NONE",
        axis=1
    )
    return df_sim


def get_classification_report(df: pd.DataFrame) -> str:
    """Genera el reporte técnico de Precisión, Recall y F1-Score a nivel fragmento."""
    y_true = df["etiqueta_real"]
    y_pred = df["pred_final"]
    report = classification_report(y_true, y_pred, zero_division=0)

    output = "=== A. EVALUACIÓN A NIVEL FRAGMENTO (Classification Report) ===\n"
    output += "Perspectiva Técnica: Evalúa la precisión exacta del modelo por fragmento.\n\n"
    output += report + "\n"
    return output


def get_recall_at_k_report(df: pd.DataFrame, id2label: dict, annotation_registry: pd.DataFrame) -> str:
    """
    Métrica de Negocio. Verifica si la cláusula real del contrato
    apareció entre los primeros K resultados ordenados por confianza matemática.
    Ignora explícitamente la clase 'NONE'.

    FIX (denominador de Recall@K, ver hilo de revisión con Gabriel Guerrero):
    El denominador ("docs_with_class") ya NO se calcula a partir de los
    fragmentos que sobrevivieron en `df` — se calcula a partir de
    `annotation_registry`, que contiene la verdad de referencia (anotaciones
    originales de CUAD) para todos los contratos efectivamente evaluados.
    Así, un contrato cuya cláusula real no fue capturada por ningún fragmento
    (por fallo de la fragmentación, no del modelo) sigue contando en el
    denominador y penaliza el recall, en vez de desaparecer del cálculo.

    El numerador (búsqueda de la clase en el Top-1/Top-3 por score dentro de
    los fragmentos de ese documento) no cambia: sigue operando sobre `df`.

    Args:
        df: ground_truth_results.csv (fragmentos con etiqueta_real y scores).
        id2label: diccionario de clases del experimento (baseline o balanced).
        annotation_registry: DataFrame ["id_documento", "categoria"] con la
                              verdad de referencia, generado en
                              03_calculate_ground_truth.py vía build_annotation_registry().
    """
    output = "=== B. EVALUACIÓN A NIVEL CONTRATO (Recall@K) ===\n"
    output += "Perspectiva Negocio: ¿La cláusula real está en el Top K de sugerencias?\n\n"

    valid_classes = [label.upper().strip() for label in id2label.values()]

    for cls in valid_classes:
        # Encontramos cómo se llama la columna de probabilidad para esta clase
        var_under = f"score_{cls.lower().replace(' ', '_').replace('-', '_')}"
        col_score = var_under if var_under in df.columns else f"score_{cls.lower().replace(' ', '_')}"

        # FIX: contratos que REALMENTE tienen esta cláusula según CUAD,
        # no según qué fragmentos sobrevivieron en df.
        docs_with_class = annotation_registry[
            annotation_registry["categoria"] == cls
        ]["id_documento"].unique()

        if len(docs_with_class) == 0:
            continue

        hits_at_1, hits_at_3 = 0, 0
        for doc in docs_with_class:
            doc_frags = df[df["id_documento"] == doc].copy()

            # Si el documento no generó ningún fragmento (o no tiene la columna
            # de score), no hay nada que pueda estar en el Top-K: cuenta como
            # fallo implícito, no se suma a hits_at_1/hits_at_3 y se continúa.
            if doc_frags.empty or col_score not in doc_frags.columns:
                continue

            # Ordenamos y tomamos el Top
            top_frags = doc_frags.sort_values(by=col_score, ascending=False)
            top_1 = top_frags.head(1)["etiqueta_real"].values
            top_3 = top_frags.head(3)["etiqueta_real"].values

            if cls in top_1: hits_at_1 += 1
            if cls in top_3: hits_at_3 += 1

        r_at_1 = (hits_at_1 / len(docs_with_class)) * 100
        r_at_3 = (hits_at_3 / len(docs_with_class)) * 100
        output += f"Categoría: {cls} (En {len(docs_with_class)} documentos)\n"
        output += f"  - Recall@1 : {r_at_1:.1f}%\n"
        output += f"  - Recall@3 : {r_at_3:.1f}%\n"

    return output + "\n"


def get_confidence_distribution_report(df: pd.DataFrame) -> str:
    """
    Métrica de Confianza Plena para el modelo Balanceado.
    Calcula la seguridad media (en porcentaje) con la que el modelo predice cada clase.
    """
    output = "=== C. DISTRIBUCIÓN DE CONFIANZA PROMEDIO ===\n"
    output += "Evalúa qué tan seguro está el modelo al hacer sus predicciones.\n\n"

    # Agrupamos por la predicción final y promediamos la confianza cruda
    confianza = df.groupby("pred_final")["confidence_raw"].mean().reset_index()
    confianza.rename(columns={"pred_final": "Clase Predicha", "confidence_raw": "Confianza Media"}, inplace=True)

    # Formato visual
    confianza["Confianza Media"] = (confianza["Confianza Media"] * 100).round(2).astype(str) + "%"
    output += confianza.to_string(index=False) + "\n\n"

    return output