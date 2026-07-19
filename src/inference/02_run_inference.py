"""
02_run_inference.py

Script orquestador de la Fase 2 (Inferencia Enriquecida).
Procesa de forma lineal y masiva el archivo 'batch_fragments.csv' mediante BERT,
calcula las probabilidades individuales por clase, aplica el umbral de confianza
de negocio (75%) y genera el dataset enriquecido final.

Al concluir con éxito, realiza un Commit masivo en el Registro Maestro pasando
los documentos procesados al estado definitivo 'Clasificado'.
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 1. Asegurar la inclusión de la raíz del proyecto en el PATH del sistema
sys.path.append(str(Path(__file__).resolve().parents[2]))

# Importación de rutas globales de configuración
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

# Importación de estándares técnicos y flujos
from src.config.text_standards import (
    BATCH_SIZE_INFERENCE,
    CONFIDENCE_THRESHOLD,
    DOC_STATES
)

# Importación del mapeo oficial de categorías e infraestructura modular
from src.config.label_mapping import ID2LABEL
from src.inference.inference_dataset import BERTInferenceDataset


def main(model_type):
    print(f"Iniciando Inferencia. Modo seleccionado: {model_type.upper()}")

    # ==========================================
    # 1. ENRUTAMIENTO DINÁMICO (BASELINE POR DEFECTO)
    # ==========================================
    if model_type == "balanced":
        print("-> Cargando carril BALANCEADO")
        registry_path = MASTER_REGISTRY_BALANCED
        fragments_path = FRAGMENTS_BALANCED
        results_path = SCORED_RESULTS_BALANCED
        model_path = MODEL_BERT_BALANCED_DIR
    else:
        # POR DEFECTO: BASELINE
        print("-> Cargando carril BASELINE (Default)")
        registry_path = MASTER_REGISTRY_BASELINE
        fragments_path = FRAGMENTS_BASELINE
        results_path = SCORED_RESULTS_BASELINE
        model_path = MODEL_BERT_BASELINE_DIR

    # ==========================================
    # 2. VALIDACIÓN DE ARCHIVOS
    # ==========================================
    if not fragments_path.exists():
        print(f"Error: No se encontró el archivo de fragmentos en {fragments_path}")
        print(f"Por favor, ejecuta primero: python src/inference/01_prepare_batch_documents.py --model {model_type}")
        return
    
    # ==========================================
    # 3. CARGA DE MODELO Y TOKENIZADOR
    # ==========================================
    print(f"Cargando modelo y tokenizador desde: {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return
    
    # Mover a GPU si está disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"[*] Entorno de hardware detectado: {device}")

    # ==========================================
    # 4. EJECUCIÓN DE INFERENCIA
    # ==========================================
    print(f"[*] Cargando fragmentos crudos desde: {fragments_path.name}...")
    df_fragments = pd.read_csv(fragments_path)
    
    if df_fragments.empty:
        print("[AVISO] El archivo de fragmentos está vacío. No hay datos para procesar.")
        return
    print(f"Procesando {len(df_fragments)} fragmentos...")

    # 5. Configuración de Seguridad para el Entorno de Inferencia
    model.eval()  # Desactiva capas de Dropout y normalizaciones de entrenamiento
    
    # 6. Preparación de la Infraestructura de Datos (Uso de clase modular)
    # Pasamos la columna 'contenido' de forma lineal (shuffle=False para garantizar orden)
    dataset = BERTInferenceDataset(
        texts=df_fragments["contenido"].tolist(),
        tokenizer=tokenizer
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE_INFERENCE, shuffle=False)

    # 7. Bucle Principal de Inferencia (Batch Processing)
    print(f"[*] Ejecutando predicciones en lotes de {BATCH_SIZE_INFERENCE}...")
    all_probabilities = []

    # context manager para apagar el cálculo de gradientes y ahorrar memoria RAM/VRAM
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Enviar tensores empaquetados al hardware correspondiente (GPU o CPU)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Ejecución matemática del modelo
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Aplicación de Softmax para transformar valores crudos (logits) en porcentajes probables
            probs = torch.softmax(logits, dim=1)
            
            # Mover tensores devueltos a la CPU y convertirlos a formato NumPy
            all_probabilities.append(probs.cpu().numpy())

    # Consolidar todos los bloques de matrices en una sola matriz gigante de NumPy
    probabilities_matrix = np.vstack(all_probabilities)

    # 8. Enriquecimiento del DataFrame (Inyección de métricas y scores de IA)
    print("[*] Enriqueciendo la estructura de datos con los resultados del modelo...")
    
    # Crear dinámicamente las columnas de los porcentajes por cada categoría del mapa oficial
    for idx, label_name in ID2LABEL.items():
        column_name = f"score_{label_name.lower().replace(' ', '_')}"
        df_fragments[column_name] = probabilities_matrix[:, idx]

    # Extraer los índices de los valores máximos (la predicción matemática pura)
    raw_pred_ids = np.argmax(probabilities_matrix, axis=1)
    df_fragments["pred_raw"] = [ID2LABEL[pid] for pid in raw_pred_ids]
    
    # Extraer el valor del porcentaje más alto (nivel de confianza bruto de la IA)
    df_fragments["confidence_raw"] = np.max(probabilities_matrix, axis=1)

    # 9. Aplicación Rígida de las Reglas de Negocio (Filtro de Umbral del 75%)
    print(f"[*] Aplicando filtro de calidad de negocio (Umbral Mínimo: {CONFIDENCE_THRESHOLD * 100}%)...")
    
    # Marcador booleano: Verdadero si la confianza no alcanza la directriz técnica
    df_fragments["is_none_implicit"] = df_fragments["confidence_raw"] < CONFIDENCE_THRESHOLD
    
    # Asignación final: Si no supera el umbral, se degrada a 'NONE'; de lo contrario, conserva su predicción
    df_fragments["pred_final"] = np.where(
        df_fragments["is_none_implicit"],
        "NONE",
        df_fragments["pred_raw"]
    )

    # ==========================================
    # 10. GUARDADO DE RESULTADOS
    # ==========================================
    print(f"[*] Guardando resultados enriquecidos en: {results_path}...")
    try:
        # Asegurar creación del directorio destino por seguridad
        results_path.parent.mkdir(parents=True, exist_ok=True)
        df_fragments.to_csv(results_path, index=False, encoding="utf-8")
        print(f"Resultados de inferencia guardados en: {results_path}")
    except Exception as e:
        print(f"[ERROR] Error crítico al escribir los resultados en disco: {e}")
        return

    # 11. Commit Final en el Registro Maestro de Documentos (Bulk Update)
    print(f"[*] Actualizando estados en el Registro Maestro: {registry_path.name}...")
    if registry_path.exists():
        try:
            df_registry = pd.read_csv(registry_path)
            
            # Identificar qué documentos acabamos de procesar
            processed_document_ids = df_fragments["id_documento"].unique()
            
            # Pasar de IN_PROGRESS a SCORED
            mask = df_registry["id_documento"].isin(processed_document_ids) & (df_registry['estado'] == DOC_STATES["IN_PROGRESS"])         
            df_registry.loc[mask, "estado"] = DOC_STATES["SCORED"]
            
            # Guardar el registro actualizado
            df_registry.to_csv(registry_path, index=False, encoding="utf-8")

            num_actualizados = mask.sum()
            print(f"Éxito. {num_actualizados} documentos marcados como 'SCORED' en {registry_path.name}.")
            
        except Exception as e:
            print(f"[ERROR] No se pudo actualizar el registro maestro debido a: {e}")
    else:
        print(f"[WARN] No se encontró el registro maestro en la ruta esperada para su actualización.")

    # 12. Reporte Ejecutivo de Control en Consola
    total_none = (df_fragments["pred_final"] == "NONE").sum()
    total_classified = len(df_fragments) - total_none
    
    print("\n" + "=" * 50)
    print("               RESUMEN DE EJECUCIÓN")
    print("=" * 50)
    print(f" Fragmentos Totales Procesados : {len(df_fragments)}")
    print(f" Asignados a Cláusulas Reales  : {total_classified} (Confianza >= 75%)")
    print(f" Enviados a Categoría 'NONE'   : {total_none} (Baja Certeza)")
    print("=" * 50)
    print("[INFO] Pipeline de inferencia completado de forma limpia.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ejecuta la inferencia sobre fragmentos preparados.")
    
    # Hacemos el argumento OPCIONAL con default 'baseline' para mayor seguridad
    parser.add_argument(
        "--model", 
        type=str, 
        choices=["baseline", "balanced"],
        default="baseline",
        help="Selecciona el entorno del modelo (default: baseline)."
    )
    
    args = parser.parse_args()
    main(args.model)