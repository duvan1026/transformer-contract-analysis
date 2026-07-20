"""
04_generate_reports.py

Fase 3B: Generación de Inteligencia de Negocio y Reportes.
Aplica lógicas asimétricas importando herramientas especializadas.
Cumple estrictamente con la centralización de rutas (0% hardcoding).
"""

import sys
import argparse
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

# Importación estricta de rutas y nombres de archivo
from src.config.paths import (
    GROUND_TRUTH_BASELINE,
    GROUND_TRUTH_BALANCED,
    INFERENCE_BASELINE_DIR,
    INFERENCE_BALANCED_DIR,
    INFERENCE_METRICS_REPORT,
    INFERENCE_CONFUSION_MATRIX_IMAGE,
    REPORT_FULL_CONFIDENCE_NAME,
    REPORT_THRESHOLD_PREFIX
)
from src.config.text_standards import CONFIDENCE_THRESHOLDS

from src.utils.metrics_utils import apply_dynamic_threshold, get_classification_report, get_recall_at_k_report, get_confidence_distribution_report
from src.utils.visualization_utils import save_confusion_matrix_image
from src.utils.export_utils import generate_contract_jsons


def procesar_baseline(df: pd.DataFrame, out_dir: Path):
    print(f"[*] Ejecutando evaluación dinámica Baseline para umbrales: {CONFIDENCE_THRESHOLDS}")
    
    for umbral in CONFIDENCE_THRESHOLDS:
        print(f"\n -> Generando reporte para umbral: {int(umbral * 100)}%")
        
        df_sim = apply_dynamic_threshold(df, umbral)
        
        # Uso de la variable global para el prefijo de la carpeta
        nombre_carpeta = f"{REPORT_THRESHOLD_PREFIX}{int(umbral * 100)}"
        carpeta_experimento = out_dir / nombre_carpeta
        carpeta_experimento.mkdir(parents=True, exist_ok=True)
        
        reporte_final = f"REPORTE DE EVALUACIÓN BASELINE - UMBRAL: {umbral}\n\n"
        reporte_final += get_classification_report(df_sim)
        reporte_final += get_recall_at_k_report(df_sim)
        
        with open(carpeta_experimento / INFERENCE_METRICS_REPORT.name, "w", encoding="utf-8") as f:
            f.write(reporte_final)
            
        ruta_img = carpeta_experimento / INFERENCE_CONFUSION_MATRIX_IMAGE.name
        save_confusion_matrix_image(df_sim, ruta_img, f"(Umbral {int(umbral * 100)}%)")
        generate_contract_jsons(df_sim, umbral, carpeta_experimento)
        

def procesar_balanced(df: pd.DataFrame, out_dir: Path):
    print("[*] Ejecutando evaluación estática Balanced (Confianza Plena)...")
    
    df["pred_final"] = df["pred_raw"]
    
    # Uso estricto de la variable global de paths.py
    carpeta_experimento = out_dir / REPORT_FULL_CONFIDENCE_NAME
    carpeta_experimento.mkdir(parents=True, exist_ok=True)
    
    reporte_final = "REPORTE DE EVALUACIÓN BALANCED - CONFIANZA PLENA\n\n"
    reporte_final += get_classification_report(df)
    reporte_final += get_confidence_distribution_report(df)
    reporte_final += get_recall_at_k_report(df)
    
    with open(carpeta_experimento / INFERENCE_METRICS_REPORT.name, "w", encoding="utf-8") as f:
        f.write(reporte_final)
        
    ruta_img = carpeta_experimento / INFERENCE_CONFUSION_MATRIX_IMAGE.name
    save_confusion_matrix_image(df, ruta_img, "(Confianza Plena)")
    generate_contract_jsons(df, 1.0, carpeta_experimento)


def main(model_type):
    print(f"[*] Fase 3B: Inteligencia de Negocio - MODO: {model_type.upper()}")

    input_path = GROUND_TRUTH_BALANCED if model_type == "balanced" else GROUND_TRUTH_BASELINE
    out_dir = INFERENCE_BALANCED_DIR if model_type == "balanced" else INFERENCE_BASELINE_DIR

    if not input_path.exists():
        print(f"[ERROR] Archivo Ground Truth no encontrado: {input_path}")
        return

    df = pd.read_csv(input_path)

    if model_type == "baseline":
        procesar_baseline(df, out_dir)
    else:
        procesar_balanced(df, out_dir)
        
    print("\n[+] Reportes y JSONs generados con éxito.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["baseline", "balanced"], default="baseline")
    args = parser.parse_args()
    main(args.model)