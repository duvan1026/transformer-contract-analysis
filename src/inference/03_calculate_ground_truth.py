"""
03_calculate_ground_truth.py

Fase 3A: Procesamiento Espacial y Verdad Terreno.
Cruza las predicciones crudas de inferencia con el archivo CUAD original
para calcular la columna 'etiqueta_real' (Inmutabilidad de datos).

Genera además el ANNOTATION_REGISTRY: el universo de cláusulas realmente
anotadas en CUAD para los contratos efectivamente evaluados en esta corrida
(fuente correcta del denominador de Recall@K, ver 04_generate_reports.py).
"""

import sys
import argparse
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config.paths import (
    SCORED_RESULTS_BASELINE,
    SCORED_RESULTS_BALANCED,
    GROUND_TRUTH_BASELINE,
    GROUND_TRUTH_BALANCED,
    MASTER_REGISTRY_BASELINE,
    MASTER_REGISTRY_BALANCED,
    ANNOTATION_REGISTRY_BASELINE,
    ANNOTATION_REGISTRY_BALANCED,
)

from src.utils.spatial_utils import load_ground_truth, get_true_label_details, build_annotation_registry

from src.config.label_mapping import ID2LABEL_BASELINE, ID2LABEL_BALANCED

from src.config.text_standards import (
    DOC_STATES,
    EXPERIMENT_NAME_INFERENCE_BASELINE,
    EXPERIMENT_NAME_INFERENCE_BALANCED,
)


def get_evaluated_contract_ids(registry_path: Path, experiment_name: str) -> list:
    """
    Extrae del registro maestro los id_documento efectivamente evaluados
    en la corrida de inferencia actual: estado="Utilizado" (Procesado) Y
    experimento == nombre de esta corrida de inferencia específicamente.

    Se usa el registro maestro (no los fragmentos/predicciones generados)
    porque un documento puede quedar marcado como procesado en 01_prepare_batch_documents.py
    incluso si, en el caso límite, no produjo ningún fragmento válido tras el
    filtro de calidad. Ese documento debe seguir contando como "evaluado" para
    efectos del denominador de Recall@K.
    """
    if not registry_path.exists():
        print(f"[ADVERTENCIA] Registro maestro no encontrado en: {registry_path}")
        return []

    df_registry = pd.read_csv(registry_path)
    mask = (
        (df_registry["estado"] == DOC_STATES["PROCESSED"]) &
        (df_registry["experimento"] == experiment_name)
    )
    return df_registry.loc[mask, "id_documento"].tolist()


def main(model_type):
    print(f"[*] Fase 3A: Calculando Ground Truth (Cruce Espacial) - MODO: {model_type.upper()}")

    if model_type == "balanced":
        input_path = SCORED_RESULTS_BALANCED
        output_path = GROUND_TRUTH_BALANCED
        registry_path = MASTER_REGISTRY_BALANCED
        annotation_registry_path = ANNOTATION_REGISTRY_BALANCED
        experiment_name = EXPERIMENT_NAME_INFERENCE_BALANCED
        current_id2label = ID2LABEL_BALANCED
    else:
        input_path = SCORED_RESULTS_BASELINE
        output_path = GROUND_TRUTH_BASELINE
        registry_path = MASTER_REGISTRY_BASELINE
        annotation_registry_path = ANNOTATION_REGISTRY_BASELINE
        experiment_name = EXPERIMENT_NAME_INFERENCE_BASELINE
        current_id2label = ID2LABEL_BASELINE

    if not input_path.exists():
        print(f"[ERROR] Archivo de inferencia no encontrado en: {input_path}")
        return

    print(f"[*] Cargando resultados matemáticos desde: {input_path.name}")
    df = pd.read_csv(input_path)
    df["pred_raw"] = df["pred_raw"].fillna("NONE").astype(str).str.strip().str.upper()

    print("[*] Abriendo dataset original humano (CUAD_v1)...")
    annotations = load_ground_truth(current_id2label)

    print("[*] Cruzando coordenadas espaciales (start/end offsets)...")
    detalles_df = df.apply(
        lambda row: pd.Series(get_true_label_details(
            row["id_documento"],
            row["start_offset"],
            row["end_offset"],
            annotations
        )), axis=1
    )
    df = pd.concat([df, detalles_df], axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"[+] Éxito: Columna 'etiqueta_real' calculada correctamente (sin circularidad con pred_raw).")
    print(f"[+] Archivo de evaluación inmutable guardado en: {output_path.name}")

    print("[*] Construyendo registro de anotaciones (denominador correcto de Recall@K)...")
    contract_ids = get_evaluated_contract_ids(registry_path, experiment_name)

    if not contract_ids:
        print("[ADVERTENCIA] No se encontraron contratos evaluados en el registro maestro "
              f"para experimento='{experiment_name}'. El annotation_registry quedará vacío.")

    df_annotation_registry = build_annotation_registry(
        annotations=annotations,
        contract_ids=contract_ids,
        valid_classes=list(current_id2label.values())
    )

    annotation_registry_path.parent.mkdir(parents=True, exist_ok=True)
    df_annotation_registry.to_csv(annotation_registry_path, index=False, encoding="utf-8")

    print(f"[+] Registro de anotaciones guardado en: {annotation_registry_path.name} "
          f"({len(contract_ids)} contratos evaluados, {len(df_annotation_registry)} pares documento-categoría).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["baseline", "balanced"], default="baseline")
    args = parser.parse_args()
    main(args.model)