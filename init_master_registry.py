import pandas as pd
import json
import os

from src.config.paths import CUAD_JSON, MASTER_DOCUMENT_REGISTRY

from src.config.paths import (
    TRAIN_DATASET, 
    VAL_DATASET, 
    TEST_DATASET
)

def init_registry():
    print("Iniciando la creación del Registro Maestro...")
    
    # 1. Obtener TODOS los documentos que existen en CUAD
    with open(CUAD_JSON, 'r', encoding='utf-8') as f:
        cuad_data = json.load(f)
    
    all_documents = []
    for doc in cuad_data['data']:
        all_documents.append(doc['title']) # 'title' es el ID en CUAD
        
    # Crear el DataFrame base con todos los documentos disponibles
    df_registry = pd.DataFrame({
        'id_documento': all_documents,
        'estado': 'Disponible',
        'experimento': 'Ninguno',
        'split_asignado': 'Ninguno'
    })
    
    print(f"Total de documentos encontrados en CUAD: {len(df_registry)}")

    # 2. Rastrear los que se usaron en la V1
    splits = {
        'train': TRAIN_DATASET,
        'validation': VAL_DATASET,
        'test': TEST_DATASET
    }
    
    for split_name, split_path in splits.items():
        if os.path.exists(split_path):
            df_split = pd.read_csv(split_path)
            # Asegúrate de usar el nombre correcto de la columna de ID (aquí asumo 'id_documento')
            col_id = 'id_documento' if 'id_documento' in df_split.columns else df_split.columns[0]
            used_ids = df_split[col_id].unique()
            
            # Actualizar el registro maestro para estos IDs
            mask = df_registry['id_documento'].isin(used_ids)
            df_registry.loc[mask, 'estado'] = 'Utilizado'
            df_registry.loc[mask, 'experimento'] = 'v1_spans_5_clases'
            df_registry.loc[mask, 'split_asignado'] = split_name
            
            print(f"- Marcados {len(used_ids)} documentos como usados en V1 ({split_name}).")
        else:
            print(f"Advertencia: No se encontró el split {split_path}")

    # 3. Guardar el Registro Maestro
    df_registry.to_csv(MASTER_DOCUMENT_REGISTRY, index=False)
    print(f"\n¡Registro Maestro creado con éxito en: {MASTER_DOCUMENT_REGISTRY}!")
    
    # Mostrar un pequeño resumen
    print("\nResumen actual del inventario:")
    print(df_registry['estado'].value_counts())

if __name__ == "__main__":
    init_registry()