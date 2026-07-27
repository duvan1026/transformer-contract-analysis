"""
01_prepare_batch_documents.py

Script orquestador de la Fase 2 (Inferencia).
Se encarga de seleccionar un lote de documentos (ej. 100 contratos),
extraer su texto original y utilizar `fragmentation_utils` para segmentarlos
en fragmentos aptos para el modelo BERT, conservando la trazabilidad.
"""
import pandas as pd
import sys
import argparse
from pathlib import Path
import json
from transformers import AutoTokenizer

# Añadir raíz del proyecto
sys.path.append(str(Path(__file__).resolve().parents[2]))

# Importar las rutas globales de tu configuración
from src.config.paths import (
    RAW_DATA_DIR, 
    CUAD_JSON,
    MASTER_REGISTRY_BASELINE,
    MASTER_REGISTRY_BALANCED,
    INFERENCE_BASELINE_DIR,
    INFERENCE_BALANCED_DIR,
    FRAGMENTS_BASELINE,
    FRAGMENTS_BALANCED,
    REJECTED_AUDIT_BASELINE,
    REJECTED_AUDIT_BALANCED
)

from src.config.text_standards import (
    RANDOM_SEED,
    MODEL_NAME,
    DOC_STATES,
    BATCH_SIZE_DOCS,
    EXPERIMENT_NAME_INFERENCE_BASELINE,
    EXPERIMENT_NAME_INFERENCE_BALANCED
)

# Importar el motor de fragmentación
from src.utils.fragmentation_utils import fragment_single_document

def main(model_type):
    print(f"Iniciando la preparación del lote de documentos para inferencia ({model_type.upper()})...")

    # 1. Enrutar las variables según el modelo seleccionado
    if model_type == "balanced":
        registry_path = MASTER_REGISTRY_BALANCED
        inference_dir = INFERENCE_BALANCED_DIR
        fragments_path = FRAGMENTS_BALANCED
        rejected_path = REJECTED_AUDIT_BALANCED
        experiment_name = EXPERIMENT_NAME_INFERENCE_BALANCED
    else:
        registry_path = MASTER_REGISTRY_BASELINE
        inference_dir = INFERENCE_BASELINE_DIR
        fragments_path = FRAGMENTS_BASELINE
        rejected_path = REJECTED_AUDIT_BASELINE
        experiment_name = EXPERIMENT_NAME_INFERENCE_BASELINE

    # 2. Crear el directorio de salida aislado si no existe
    inference_dir.mkdir(parents=True, exist_ok=True)

    # 3. Cargar el Tokenizador
    print(f"Cargando tokenizador: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 4. Seleccionar y Reservar los documentos a procesar desde el registro correcto
    try:
        df_registry = pd.read_csv(registry_path)
        
        mask_disponibles = (
            (df_registry['estado'] == DOC_STATES["AVAILABLE"]) & 
            (df_registry['experimento'] == DOC_STATES["NONE"]) & 
            (df_registry['split_asignado'] == DOC_STATES["NONE"])
        )
        
        df_disponibles = df_registry[mask_disponibles]
        num_docs_a_tomar = min(BATCH_SIZE_DOCS, len(df_disponibles))
        
        if num_docs_a_tomar == 0:
            print("No hay documentos con estado 'Disponible' en el registro {registry_path.name}.")
            return
            
        df_batch = df_disponibles.sample(n=num_docs_a_tomar, random_state=RANDOM_SEED)
        document_ids = df_batch['id_documento'].tolist()
        
        print(f"Se seleccionaron {len(document_ids)} documentos. Reservándolos...")
        
        # Estado intermedio "En_proceso"
        df_registry.loc[df_batch.index, 'estado'] = DOC_STATES["IN_PROGRESS"]
        df_registry.to_csv(registry_path, index=False, encoding="utf-8")
        
    except FileNotFoundError:
        print(f"Error crítico: No se encontró el registro en {registry_path}.")
        print(f"Ejecuta primero: python src/data/00_init_master_registry.py --model {model_type}")
        return
    
    # 5. Procesamiento en lote (Batch Processing)
    all_fragments = []
    all_rejected = []
    processed_ids = [] # Lista para rastrear los que SÍ fueron exitosos
    
    cuad_json_path = CUAD_JSON
    
    try:
        with open(cuad_json_path, "r", encoding="utf-8") as f:
            cuad_data = json.load(f)["data"]
    except FileNotFoundError:
        print(f"Error: No se encontró el dataset original en {cuad_json_path}")
        # Si falla aquí, los documentos se quedan como "En_proceso" para revisión manual
        return

    print("Fragmentando documentos...")
    cuad_dict = {item["title"]: item for item in cuad_data}

    for doc_id in document_ids:
        if doc_id in cuad_dict:
            raw_text = cuad_dict[doc_id]["paragraphs"][0]["context"]
            
            # Ejecutamos el motor de fragmentación 
            valid_docs, rejected_docs = fragment_single_document(
                document_id=doc_id, 
                raw_text=raw_text, 
                tokenizer=tokenizer
            )
            
            all_fragments.extend(valid_docs)
            all_rejected.extend(rejected_docs)
            
            # Anotamos como exitoso solo si la fragmentación no falló
            processed_ids.append(doc_id)
        else:
            print(f"  - Documento {doc_id} no encontrado en CUAD_v1.json. Revirtiendo estado.")
            # Si no se encontró, lo devolvemos a Disponible inmediatamente en memoria
            df_registry.loc[df_registry['id_documento'] == doc_id, 'estado'] = DOC_STATES["AVAILABLE"]

    # 6. Guardar los resultados y hacer el Commit Final
    if all_fragments:
        df_fragments = pd.DataFrame(all_fragments)
        df_rejected = pd.DataFrame(all_rejected)
        
        try:
            # Guardamos los fragmentos en la ruta aislada del modelo
            df_fragments.to_csv(fragments_path, index=False, encoding="utf-8")
            df_rejected.to_csv(rejected_path, index=False, encoding="utf-8")
            
            # Marcamos los procesados y asignamos el experimento dinámico
            mask_processed = df_registry['id_documento'].isin(processed_ids)
            df_registry.loc[mask_processed, 'estado'] = DOC_STATES["PROCESSED"]
            df_registry.loc[mask_processed, 'experimento'] = experiment_name
            
            # Guardamos el estado definitivo en el CSV Maestro
            df_registry.to_csv(registry_path, index=False, encoding="utf-8")
            
            print(f"Registro maestro actualizado. {len(processed_ids)} documentos marcados como 'Utilizado'.")
            print(f"Resultados guardados en: {inference_dir.name}/")
            
        except Exception as e:
            print(f"Error al guardar los archivos: {e}")
            print("Los documentos permanecerán en 'En_proceso' en el registro maestro.")
    else:
        print("\nNo se generó ningún fragmento. Revisa los datos de entrada.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepara lotes de documentos para inferencia.")
    parser.add_argument(
        "--model", 
        type=str, 
        choices=["baseline", "balanced"],
        required=True,
        help="Selecciona el entorno del modelo: 'baseline' o 'balanced'."
    )
    
    args = parser.parse_args()
    main(args.model)