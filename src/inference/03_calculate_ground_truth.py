"""
03_calculate_ground_truth.py

Fase 3A: Procesamiento Espacial y Verdad Terreno.
Cruza las predicciones crudas de inferencia con el archivo CUAD original
para calcular la columna 'etiqueta_real' (Inmutabilidad de datos).
"""

import sys
import argparse
import pandas as pd
from pathlib import Path

# 1. Asegurar la inclusión de la raíz del proyecto en el PATH
sys.path.append(str(Path(__file__).resolve().parents[2]))

# Importar rutas de forma centralizada (SIN RUTAS QUEMADAS)
from src.config.paths import (
    SCORED_RESULTS_BASELINE,
    SCORED_RESULTS_BALANCED,
    GROUND_TRUTH_BASELINE,
    GROUND_TRUTH_BALANCED
)

# 2. Importar herramientas espaciales 
from src.utils.spatial_utils import load_ground_truth, get_true_label_for_fragment

def main(model_type):
    print(f"[*] Fase 3A: Calculando Ground Truth (Cruce Espacial) - MODO: {model_type.upper()}")

    # ==========================================
    # ENRUTAMIENTO DINÁMICO DESDE paths.py
    # ==========================================
    if model_type == "balanced":
        input_path = SCORED_RESULTS_BALANCED
        output_path = GROUND_TRUTH_BALANCED
    else:
        input_path = SCORED_RESULTS_BASELINE
        output_path = GROUND_TRUTH_BASELINE

    if not input_path.exists():
        print(f"[ERROR] Archivo de inferencia no encontrado en: {input_path}")
        return

    # ==========================================
    # CARGA DE DATOS
    # ==========================================
    print(f"[*] Cargando resultados matemáticos desde: {input_path.name}")
    df = pd.read_csv(input_path)
    
    # Rellenar nulos con NONE y estandarizar la predicción cruda
    df["pred_raw"] = df["pred_raw"].fillna("NONE").astype(str).str.strip().str.upper()

    print("[*] Abriendo dataset original humano (CUAD_v1)...")
    annotations = load_ground_truth()

    # ==========================================
    # CÁLCULO MATEMÁTICO DE SUPERPOSICIÓN
    # ==========================================
    print("[*] Cruzando coordenadas espaciales (start/end offsets)...")
    df["etiqueta_real"] = df.apply(
        lambda row: get_true_label_for_fragment(
            row["id_documento"], 
            row["start_offset"], 
            row["end_offset"], 
            row["pred_raw"], 
            annotations
        ), axis=1
    )

    # ==========================================
    # GUARDADO Y CONFIRMACIÓN
    # ==========================================
    # Se genera un archivo nuevo para no destruir el estado puro de la inferencia
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    
    print(f"[+] Éxito: Columna 'etiqueta_real' calculada correctamente.")
    print(f"[+] Archivo de evaluación inmutable guardado en: {output_path.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["baseline", "balanced"], default="baseline")
    args = parser.parse_args()
    main(args.model)