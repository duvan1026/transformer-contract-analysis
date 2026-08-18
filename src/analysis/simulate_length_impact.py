"""
simulate_length_impact.py

Estima el impacto de que la longitud de los fragmentos de inferencia se
pareciera a la de entrenamiento: compara, por clase, el recall/especificidad
ponderado bajo la distribucion real actual vs. bajo la distribucion de train
(escenario objetivo), y deriva macro-recall y accuracy globales estimados.

Independiente de analyze_recall_by_length.py y de analyze_token_length.py:
no requiere que se hayan ejecutado antes ni depende de sus salidas. Es un
paso opcional y mas costoso que los otros dos (carga tokenizer y tokeniza
train.csv completo), por eso vive en su propio script en vez de anidarse
en analyze_recall_by_length.py.

Toda la logica de carga, calculo y graficado vive en
src/utils/recall_by_length/ y src/utils/token_length/. Este script solo
orquesta: CLI, enrutamiento por --model, flujo y guardado.

Limitacion explicita: NO estima macro-F1 (requiere precision por clase,
no disponible en el ground truth agregado). Es un techo estimado bajo el
supuesto de que la relacion longitud-acierto observada hoy se mantiene
con fragmentos nuevos mas cortos; no reemplaza una validacion real con
reentrenamiento.

Uso:
    python src/analysis/simulate_length_impact.py --model baseline
    python src/analysis/simulate_length_impact.py --model balanced
"""

import sys
import argparse
from pathlib import Path

# Añadir raíz del proyecto (debe ir antes de los imports de src.*)
sys.path.append(str(Path(__file__).resolve().parents[2]))

from transformers import AutoTokenizer

from src.config.paths import (
    GROUND_TRUTH_BASELINE,
    GROUND_TRUTH_BALANCED,
    TRAIN_DATASET,
    RECALL_BY_LENGTH_BASELINE_DIR,
    RECALL_BY_LENGTH_BALANCED_DIR,
    IMPACT_SIMULATION_STATS_BASELINE,
    IMPACT_SIMULATION_CHART_BASELINE,
    IMPACT_SIMULATION_STATS_BALANCED,
    IMPACT_SIMULATION_CHART_BALANCED,
)
from src.config.text_standards import RECALL_BY_LENGTH_BINS, MODEL_NAME
from src.config.label_mapping import LABEL2ID_BASELINE, LABEL2ID_BALANCED

from src.utils.recall_by_length.io import cargar_ground_truth_positivos, cargar_ground_truth_none
from src.utils.recall_by_length.stats import calcular_recall_por_bin
from src.utils.recall_by_length.simulation import construir_tabla_impacto
from src.utils.recall_by_length.plots import graficar_impacto_estimado
from src.utils.token_length.io import cargar_split


def main(model_type):
    print(f"Iniciando simulacion de impacto por longitud ({model_type.upper()})...")
    print("Independiente de analyze_recall_by_length.py y analyze_token_length.py "
          "(no requiere que se hayan ejecutado antes).\n")

    if model_type == "balanced":
        ground_truth_path = GROUND_TRUTH_BALANCED
        output_dir = RECALL_BY_LENGTH_BALANCED_DIR
        impact_stats_path = IMPACT_SIMULATION_STATS_BALANCED
        impact_chart_path = IMPACT_SIMULATION_CHART_BALANCED
        etiquetas_validas = set(LABEL2ID_BALANCED.keys())
    else:
        ground_truth_path = GROUND_TRUTH_BASELINE
        output_dir = RECALL_BY_LENGTH_BASELINE_DIR
        impact_stats_path = IMPACT_SIMULATION_STATS_BASELINE
        impact_chart_path = IMPACT_SIMULATION_CHART_BASELINE
        etiquetas_validas = set(LABEL2ID_BASELINE.keys())

    output_dir.mkdir(parents=True, exist_ok=True)

    if not ground_truth_path.exists():
        print(f"[ERROR] No se encontro {ground_truth_path}. "
              f"Este analisis requiere el ground truth de inferencia ya generado.")
        return

    print("Cargando ground truth (spans positivos y NONE)...")
    df_positivos = cargar_ground_truth_positivos(ground_truth_path, columna_prediccion="pred_raw")
    df_none = cargar_ground_truth_none(ground_truth_path, columna_prediccion="pred_raw")

    if df_positivos.empty:
        print("[ERROR] No se encontraron spans positivos validos en el ground truth. Abortando.")
        return

    print("Calculando curvas de recall (positivas) y especificidad (NONE) por longitud...")
    df_recall = calcular_recall_por_bin(df_positivos, RECALL_BY_LENGTH_BINS)
    df_especificidad = calcular_recall_por_bin(df_none, RECALL_BY_LENGTH_BINS)

    print(f"Cargando tokenizador: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Tokenizando train.csv para el escenario objetivo...")
    df_train = cargar_split(TRAIN_DATASET, "train", "span_text", "categoria", etiquetas_validas, tokenizer, [])

    distribuciones = {}
    for clase in sorted(etiquetas_validas - {"none"}):
        distribuciones[clase] = {
            "actual": df_positivos.loc[df_positivos["clase"] == clase, "tokens_estimados"],
            "objetivo": df_train.loc[df_train["clase"] == clase, "n_tokens"] if not df_train.empty else None,
        }
    if "none" in etiquetas_validas:
        distribuciones["none"] = {
            "actual": df_none["tokens_estimados"] if not df_none.empty else None,
            "objetivo": df_train.loc[df_train["clase"] == "none", "n_tokens"] if not df_train.empty else None,
        }

    print("Calculando escenario actual vs. objetivo...")
    tabla_impacto, resumen_global = construir_tabla_impacto(
        df_recall, df_especificidad, distribuciones, RECALL_BY_LENGTH_BINS
    )
    print(tabla_impacto.to_string(index=False))
    print(f"\nMacro-Recall (positivas): {resumen_global['macro_recall_actual']}% -> "
          f"{resumen_global['macro_recall_objetivo']}%")
    print(f"Accuracy global estimada: {resumen_global['accuracy_actual']}% -> "
          f"{resumen_global['accuracy_objetivo']}%")
    print("Nota: Macro-F1 no se estima aqui — requiere precision por clase, no disponible "
          "en el ground truth agregado. Este resultado es un techo estimado, valido solo "
          "bajo el supuesto de que la relacion longitud-acierto se mantiene con fragmentos nuevos; "
          "no reemplaza una validacion real con reentrenamiento.")

    tabla_impacto.to_excel(impact_stats_path, index=False, sheet_name="simulacion_impacto")
    graficar_impacto_estimado(tabla_impacto, resumen_global, impact_chart_path, title_suffix=f"({model_type})")
    print(f"\nSimulacion de impacto guardada en {impact_stats_path} y {impact_chart_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simula el impacto en recall/especificidad/accuracy de igualar la longitud "
                     "de inferencia a la de train."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["baseline", "balanced"],
        required=True,
        help="Selecciona el carril de inferencia: 'baseline' o 'balanced'.",
    )
    args = parser.parse_args()
    main(args.model)