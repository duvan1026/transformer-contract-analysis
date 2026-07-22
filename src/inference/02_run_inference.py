"""
02_run_inference.py

Script orquestador de la Fase 2 (Inferencia Pura).
Actúa exclusivamente como motor matemático: procesa masivamente los fragmentos,
calcula la matriz de probabilidades (Softmax) y extrae la predicción cruda (argmax).

ATENCIÓN: Las reglas de negocio y umbrales han sido delegadas a la Fase 3 (Evaluación)
para permitir simulaciones eficientes sin reprocesamiento neuronal.
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 1. Asegurar la inclusión de la raíz del proyecto en el PATH
sys.path.append(str(Path(__file__).resolve().parents[2]))

# Importación de rutas globales
from src.config.paths import (
    MASTER_REGISTRY_BASELINE,
    MASTER_REGISTRY_BALANCED,
    FRAGMENTS_BASELINE,
    FRAGMENTS_BALANCED,
    SCORED_RESULTS_BASELINE,
    SCORED_RESULTS_BALANCED,
    MODEL_BERT_BASELINE_DIR,
    MODEL_BERT_BALANCED_DIR
)

# Importación de estándares técnicos
from src.config.text_standards import (
    BATCH_SIZE_INFERENCE,
    DOC_STATES
)

# Importar los diccionarios específicos en lugar del genérico
from src.config.label_mapping import ID2LABEL_BASELINE, ID2LABEL_BALANCED
# Importación de infraestructura modular
from src.inference.inference_dataset import BERTInferenceDataset


def main(model_type):
    print(f"[*] Iniciando Inferencia Pura. MODO: {model_type.upper()}")

    # ==========================================
    # 1. ENRUTAMIENTO DINÁMICO
    # ==========================================
    if model_type == "balanced":
        registry_path = MASTER_REGISTRY_BALANCED
        fragments_path = FRAGMENTS_BALANCED
        results_path = SCORED_RESULTS_BALANCED
        model_path = MODEL_BERT_BALANCED_DIR
        current_id2label = ID2LABEL_BALANCED
    else:
        registry_path = MASTER_REGISTRY_BASELINE
        fragments_path = FRAGMENTS_BASELINE
        results_path = SCORED_RESULTS_BASELINE
        model_path = MODEL_BERT_BASELINE_DIR
        current_id2label = ID2LABEL_BASELINE

    # ==========================================
    # 2. VALIDACIÓN DE ARCHIVOS
    # ==========================================
    if not fragments_path.exists():
        print(f"[ERROR] No se encontró el archivo de fragmentos en {fragments_path}")
        return
    
    # ==========================================
    # 3. CARGA DE MODELO Y EXTRACCIÓN DINÁMICA DE CLASES
    # ==========================================
    print(f"[*] Cargando modelo desde: {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    except Exception as e:
        print(f"[ERROR] Falló la carga del modelo: {e}")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"[*] Hardware activo: {device}")

    # Usar la ÚNICA FUENTE DE VERDAD (label_mapping) ignorando la configuración defectuosa del modelo
    model_id2label = {int(k): str(v).strip().upper() for k, v in current_id2label.items()}
    print(f"[*] Clases detectadas nativamente: {list(model_id2label.values())}")

    # ==========================================
    # 4. EJECUCIÓN DE INFERENCIA NEURONAL
    # ==========================================
    df_fragments = pd.read_csv(fragments_path)
    if df_fragments.empty:
        print("[AVISO] El archivo de fragmentos está vacío.")
        return
        
    model.eval()
    dataset = BERTInferenceDataset(texts=df_fragments["contenido"].tolist(), tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE_INFERENCE, shuffle=False)

    print(f"[*] Ejecutando propagación hacia adelante (Batch: {BATCH_SIZE_INFERENCE})...")
    all_probabilities = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            all_probabilities.append(probs.cpu().numpy())

    probabilities_matrix = np.vstack(all_probabilities)

    # ==========================================
    # 5. ENRIQUECIMIENTO MATEMÁTICO DEL DATASET
    # ==========================================
    print("[*] Inyectando métricas crudas en el dataset...")
    
    # Probabilidades individuales por clase
    for idx, label_name in model_id2label.items():
        column_name = f"score_{label_name.lower().replace(' ', '_').replace('-', '_')}"
        df_fragments[column_name] = probabilities_matrix[:, idx]

    # Predicción absoluta (argmax) y su porcentaje
    raw_pred_ids = np.argmax(probabilities_matrix, axis=1)
    df_fragments["pred_raw"] = [model_id2label[pid] for pid in raw_pred_ids]
    df_fragments["confidence_raw"] = np.max(probabilities_matrix, axis=1)

    # ==========================================
    # 6. GUARDADO DE RESULTADOS Y COMMIT
    # ==========================================
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df_fragments.to_csv(results_path, index=False, encoding="utf-8")
    print(f"[*] Resultados guardados en: {results_path}")
    
    if registry_path.exists():
        df_registry = pd.read_csv(registry_path)
        processed_ids = df_fragments["id_documento"].unique()
        mask = df_registry["id_documento"].isin(processed_ids) & (df_registry['estado'] == DOC_STATES["IN_PROGRESS"])         
        df_registry.loc[mask, "estado"] = DOC_STATES["SCORED"]
        df_registry.to_csv(registry_path, index=False, encoding="utf-8")
        print(f"[+] Trazabilidad: {mask.sum()} documentos marcados como 'SCORED'.")

    print("[INFO] Motor de inferencia completado.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["baseline", "balanced"], default="baseline")
    args = parser.parse_args()
    main(args.model)