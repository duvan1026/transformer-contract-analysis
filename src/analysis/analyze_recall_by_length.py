"""
analyze_recall_by_length.py

Verifica, con el ground truth real, si la zona de menor acierto del modelo
detectada previamente (el rango 400-600 tokens) sigue siendo valida —
en vez de asumirla como un valor fijo heredado de un analisis anterior.

Este script se ejecuta ANTES de analyze_token_length.py: calcula la zona
de riesgo real y la guarda en un archivo (zona_riesgo_calculada.json) que
analyze_token_length.py lee automaticamente. Si no se ejecuta primero,
analyze_token_length.py usa el valor de respaldo en text_standards.py y
lo advierte por consola.

Toda la logica de carga, calculo y graficado vive en
src/utils/recall_by_length/ (io.py, stats.py, plots.py). Este script solo
orquesta: CLI, enrutamiento por --model, flujo y guardado.

Uso:
    python src/analysis/analyze_recall_by_length.py --model baseline
    python src/analysis/analyze_recall_by_length.py --model balanced
"""

import sys
import json
import argparse
from pathlib import Path

# Añadir raíz del proyecto (debe ir antes de los imports de src.*)
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config.paths import (
    GROUND_TRUTH_BASELINE,
    GROUND_TRUTH_BALANCED,
    RECALL_BY_LENGTH_BASELINE_DIR,
    RECALL_BY_LENGTH_STATS_BASELINE,
    RECALL_BY_LENGTH_CHART_BASELINE,
    RECALL_BY_LENGTH_RISK_ZONE_FILE_BASELINE,
    RECALL_BY_LENGTH_BALANCED_DIR,
    RECALL_BY_LENGTH_STATS_BALANCED,
    RECALL_BY_LENGTH_CHART_BALANCED,
    RECALL_BY_LENGTH_RISK_ZONE_FILE_BALANCED,
)
from src.config.text_standards import RECALL_BY_LENGTH_BINS, RECALL_BY_LENGTH_MIN_N_BIN

from src.utils.recall_by_length.io import cargar_ground_truth_positivos
from src.utils.recall_by_length.stats import calcular_recall_por_bin, determinar_zona_riesgo
from src.utils.recall_by_length.plots import graficar_recall_por_longitud


def main(model_type):
    print(f"Iniciando verificacion de recall por longitud ({model_type.upper()})...")
    print("Este analisis debe ejecutarse ANTES de analyze_token_length.py: "
          "su resultado (zona de riesgo calculada) se usa automaticamente "
          "en las graficas de longitud de esa segunda corrida.\n")

    if model_type == "balanced":
        ground_truth_path = GROUND_TRUTH_BALANCED
        output_dir = RECALL_BY_LENGTH_BALANCED_DIR
        stats_path = RECALL_BY_LENGTH_STATS_BALANCED
        chart_path = RECALL_BY_LENGTH_CHART_BALANCED
        risk_zone_path = RECALL_BY_LENGTH_RISK_ZONE_FILE_BALANCED
    else:
        ground_truth_path = GROUND_TRUTH_BASELINE
        output_dir = RECALL_BY_LENGTH_BASELINE_DIR
        stats_path = RECALL_BY_LENGTH_STATS_BASELINE
        chart_path = RECALL_BY_LENGTH_CHART_BASELINE
        risk_zone_path = RECALL_BY_LENGTH_RISK_ZONE_FILE_BASELINE

    output_dir.mkdir(parents=True, exist_ok=True)

    if not ground_truth_path.exists():
        print(f"[ERROR] No se encontro {ground_truth_path}. "
              f"Este analisis requiere el ground truth de inferencia ya generado.")
        return

    print("Cargando ground truth y filtrando spans reales positivos...")
    df_positivos = cargar_ground_truth_positivos(ground_truth_path, columna_prediccion="pred_raw")

    if df_positivos.empty:
        print("[ERROR] No se encontraron spans positivos validos en el ground truth. Abortando.")
        return

    print(f"Spans reales positivos encontrados: {len(df_positivos)}")

    print("Calculando recall por rango de longitud...")
    df_recall = calcular_recall_por_bin(df_positivos, RECALL_BY_LENGTH_BINS)
    print(df_recall.to_string(index=False))

    zona_riesgo = determinar_zona_riesgo(df_recall, n_minimo=RECALL_BY_LENGTH_MIN_N_BIN)
    if zona_riesgo:
        print(f"\nZona de riesgo calculada: {zona_riesgo['bin_min']}-{zona_riesgo['bin_max']} tokens "
              f"(recall {zona_riesgo['recall']}%, n={zona_riesgo['n']})")
    else:
        print("\n[AVISO] No se pudo determinar una zona de riesgo (sin datos suficientes).")

    print(f"Guardando estadisticos en {stats_path}")
    df_recall.to_excel(stats_path, index=False, sheet_name="recall_por_longitud")

    print(f"Generando grafica en {chart_path}")
    graficar_recall_por_longitud(df_recall, chart_path, title_suffix=f"({model_type})", zona_riesgo=zona_riesgo)

    if zona_riesgo:
        with open(risk_zone_path, "w", encoding="utf-8") as f:
            json.dump({
                "modelo": model_type,
                "bin_min": zona_riesgo["bin_min"],
                "bin_max": zona_riesgo["bin_max"],
                "recall": zona_riesgo["recall"],
                "n": zona_riesgo["n"],
                "bins_usados": RECALL_BY_LENGTH_BINS,
                "n_minimo_por_bin": RECALL_BY_LENGTH_MIN_N_BIN,
            }, f, indent=2, ensure_ascii=False)
        print(f"Zona de riesgo guardada en {risk_zone_path} "
              f"(sera leida automaticamente por analyze_token_length.py)")

    print(f"\nVerificacion de recall por longitud ({model_type}) finalizada.")
    print(f"Siguiente paso: python src/analysis/analyze_token_length.py --model {model_type}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verifica con el ground truth real la zona de menor acierto por longitud de fragmento."
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