"""
01_prepare_batch_documents.py

Script orquestador de la Fase 2 (Inferencia).
Se encarga de seleccionar un lote de documentos (ej. 100 contratos),
extraer su texto original y utilizar `fragmentation_utils` para segmentarlos
en fragmentos aptos para el modelo BERT, conservando la trazabilidad.
"""
import pandas as pd
import sys
from pathlib import Path
import json
from transformers import AutoTokenizer

# Añadir raíz del proyecto
sys.path.append(str(Path(__file__).resolve().parents[2]))

# Importar las rutas globales de tu configuración
from src.config.paths import (
    RAW_DATA_DIR, 
    INFERENCE_DIR, 
    FRAGMENTS_DATASET,
    MASTER_DOCUMENT_REGISTRY 
)

# Importar el motor de fragmentación
from src.utils.fragmentation_utils import fragment_single_document

def main():
    print("Iniciando la preparación del lote de documentos para inferencia...")

    # 1. Crear el directorio de salida si no existe
    INFERENCE_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Cargar el Tokenizador
    model_checkpoint = "bert-base-uncased" 
    print(f"Cargando tokenizador: {model_checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # 3. Seleccionar y Reservar los documentos a procesar
    registry_path = MASTER_DOCUMENT_REGISTRY

    try:
        df_registry = pd.read_csv(registry_path)
        
        mask_disponibles = (
            (df_registry['estado'] == 'Disponible') & 
            (df_registry['experimento'] == 'Ninguno') & 
            (df_registry['split_asignado'] == 'Ninguno')
        )
        
        df_disponibles = df_registry[mask_disponibles]
        num_docs_a_tomar = min(100, len(df_disponibles))
        
        if num_docs_a_tomar == 0:
            print("No hay documentos con estado 'Disponible' en el registro.")
            return
            
        df_batch = df_disponibles.sample(n=num_docs_a_tomar, random_state=42)
        document_ids = df_batch['id_documento'].tolist()
        
        print(f"Se seleccionaron {len(document_ids)} documentos. Reservándolos...")
        
        # --- CAMBIO CLAVE 1: Estado intermedio "En_proceso" ---
        df_registry.loc[df_batch.index, 'estado'] = 'En_proceso'
        df_registry.to_csv(registry_path, index=False, encoding="utf-8")
        
    except FileNotFoundError:
        print(f"Error crítico: No se encontró el registro en {registry_path}.")
        return
    
    # 4. Procesamiento en lote (Batch Processing)
    all_fragments = []
    processed_ids = [] # Lista para rastrear los que SÍ fueron exitosos
    
    cuad_json_path = RAW_DATA_DIR / "CUAD_v1.json"
    
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
            
            # Ejecutamos el motor de fragmentación (que ahora es recursivo)
            doc_fragments = fragment_single_document(
                document_id=doc_id, 
                raw_text=raw_text, 
                tokenizer=tokenizer, 
                max_tokens=512
            )
            
            all_fragments.extend(doc_fragments)
            
            # Anotamos como exitoso solo si la fragmentación no falló
            processed_ids.append(doc_id)
        else:
            print(f"  - Documento {doc_id} no encontrado en CUAD_v1.json. Revirtiendo estado.")
            # Si no se encontró, lo devolvemos a Disponible inmediatamente en memoria
            df_registry.loc[df_registry['id_documento'] == doc_id, 'estado'] = 'Disponible'

    # 5. Guardar los resultados y hacer el Commit Final en el Registro Maestro
    if all_fragments:
        df_fragments = pd.DataFrame(all_fragments)
        
        try:
            # Primero guardamos los fragmentos físicos
            df_fragments.to_csv(FRAGMENTS_DATASET, index=False, encoding="utf-8")
            print(f"\n¡Archivo de fragmentos guardado en: {FRAGMENTS_DATASET}")
            
            # --- CAMBIO CLAVE 3: El "Commit" ---
            # Solo si se guardó el CSV correctamente, marcamos los procesados como 'Utilizado'
            mask_processed = df_registry['id_documento'].isin(processed_ids)
            df_registry.loc[mask_processed, 'estado'] = 'Utilizado'
            df_registry.loc[mask_processed, 'experimento'] = 'v2_inferencia_100_docs'
            
            # Guardamos el estado definitivo en el CSV Maestro
            df_registry.to_csv(registry_path, index=False, encoding="utf-8")
            
            print(f"Registro maestro actualizado. {len(processed_ids)} documentos marcados como 'Utilizado'.")
            
        except Exception as e:
            print(f"Error al guardar los archivos: {e}")
            print("Los documentos permanecerán en 'En_proceso' en el registro maestro.")
    else:
        print("\nNo se generó ningún fragmento. Revisa los datos de entrada.")


if __name__ == "__main__":
    main()