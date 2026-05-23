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
    FRAGMENTS_DATASET,
    INFERENCE_SCORED_RESULTS,
    MASTER_DOCUMENT_REGISTRY,
    MODEL_BERT_FINAL_DIR
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


def main():
    print("=" * 70)
    print("      INICIANDO PIPELINE DE INFERENCIA ENRIQUECIDA (BERT) - FASE 2")
    print("=" * 70)

    # 2. Selección e Inicialización de Hardware (Aceleración por GPU si está disponible)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Entorno de hardware detectado: {device}")

    # 3. Control de existencia de datos de entrada
    if not FRAGMENTS_DATASET.exists():
        print(f"[ERROR] No se encontró el archivo de fragmentos en: {FRAGMENTS_DATASET}")
        print("Asegúrate de ejecutar primero el script '01_prepare_batch_documents.py'.")
        return

    print(f"[*] Cargando fragmentos crudos desde: {FRAGMENTS_DATASET.name}...")
    df_fragments = pd.read_csv(FRAGMENTS_DATASET)
    
    if df_fragments.empty:
        print("[AVISO] El archivo de fragmentos está vacío. No hay datos para procesar.")
        return
    print(f"[+] Total de fragmentos cargados para clasificar: {len(df_fragments)}")

    # 4. Carga del "Cerebro" de la IA (Pesos del modelo BERT Final entrenado)
    print(f"[*] Cargando modelo y tokenizador desde: {MODEL_BERT_FINAL_DIR.name}...")
    print(MODEL_BERT_FINAL_DIR)
    try:
        # 1. Cargamos el tokenizador y el modelo desde la carpeta donde se guardó el mejor checkpoint
        tokenizer = AutoTokenizer.from_pretrained(MODEL_BERT_FINAL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_BERT_FINAL_DIR).to(device)
    except Exception as e:
        print(f"[ERROR] Error crítico al cargar el modelo pre-entrenado: {e}")
        return

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

    # 10. Persistencia Física del Dataset Enriquecido Final
    print(f"[*] Guardando resultados enriquecidos en: {INFERENCE_SCORED_RESULTS}...")
    try:
        # Asegurar creación del directorio destino por seguridad
        INFERENCE_SCORED_RESULTS.parent.mkdir(parents=True, exist_ok=True)
        df_fragments.to_csv(INFERENCE_SCORED_RESULTS, index=False, encoding="utf-8")
        print("[+] Archivo de resultados guardado con éxito.")
    except Exception as e:
        print(f"[ERROR] Error crítico al escribir los resultados en disco: {e}")
        return

    # 11. Commit Final en el Registro Maestro de Documentos (Bulk Update)
    print(f"[*] Actualizando estados en el Registro Maestro: {MASTER_DOCUMENT_REGISTRY.name}...")
    if MASTER_DOCUMENT_REGISTRY.exists():
        try:
            df_registry = pd.read_csv(MASTER_DOCUMENT_REGISTRY)
            
            # Obtener el listado único de los IDs de documentos que acabamos de procesar linealmente
            processed_document_ids = df_fragments["id_documento"].unique()
            
            # Identificar las filas correspondientes en el maestro
            mask = df_registry["id_documento"].isin(processed_document_ids)
            
            # Cambiar su estado masivamente de 'Utilizado' a 'Clasificado' (SCORED)
            df_registry.loc[mask, "estado"] = DOC_STATES["SCORED"]
            
            # Guardar cambios definitivos
            df_registry.to_csv(MASTER_DOCUMENT_REGISTRY, index=False, encoding="utf-8")
            print(f"[+] Éxito: {len(processed_document_ids)} contratos marcados como 'Clasificado' (SCORED).")
            
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
    main()