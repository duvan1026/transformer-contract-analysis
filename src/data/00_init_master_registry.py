import pandas as pd
import json
import os
import sys
from pathlib import Path

# 1. Asegurar la inclusión de la raíz del proyecto en el PATH del sistema
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config.paths import (
    CUAD_JSON, 
    MASTER_DOCUMENT_REGISTRY,
    TRAIN_DATASET, 
    VAL_DATASET, 
    TEST_DATASET
)

# IMPORTAMOS DICCIONARIO DE ESTADOS
from src.config.text_standards import DOC_STATES

# Centralizamos los nombres de columnas aquí mismo si no los tienes en configuración
COL_ID_DOC = 'id_documento'
COL_ESTADO = 'estado'
COL_EXPERIMENTO = 'experimento'
COL_SPLIT = 'split_asignado'
CURRENT_EXPERIMENT_NAME = 'v1_spans_5_clases'

def init_registry():
    print("Iniciando la creación del Registro Maestro...")
    
    # 1. Obtener TODOS los documentos que existen en CUAD
    with open(CUAD_JSON, 'r', encoding='utf-8') as f:
        cuad_data = json.load(f)
    
    all_documents = []
    for doc in cuad_data['data']:
        all_documents.append(doc['title']) # 'title' es el ID en CUAD

    # Crear el DataFrame base USANDO TU DICCIONARIO
    df_registry = pd.DataFrame({
        COL_ID_DOC: all_documents,
        COL_ESTADO: DOC_STATES["AVAILABLE"],       # Queda como "Disponible"
        COL_EXPERIMENTO: DOC_STATES["NONE"],       # Queda como "Ninguno"
        COL_SPLIT: DOC_STATES["NONE"]              # Queda como "Ninguno"
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

            col_id_search = COL_ID_DOC if COL_ID_DOC in df_split.columns else df_split.columns[0]
            used_ids = df_split[col_id_search].unique()

            # Actualizar el registro maestro USANDO TU DICCIONARIO
            mask = df_registry[COL_ID_DOC].isin(used_ids)
            df_registry.loc[mask, COL_ESTADO] = DOC_STATES["PROCESSED"] # Queda como "Utilizado"
            df_registry.loc[mask, COL_EXPERIMENTO] = CURRENT_EXPERIMENT_NAME
            df_registry.loc[mask, COL_SPLIT] = split_name        
            
            print(f"- Marcados {len(used_ids)} documentos como usados en V1 ({split_name}).")
        else:
            print(f"Advertencia: No se encontró el split {split_path}")


    # 3. Guardar el Registro Maestro
    df_registry.to_csv(MASTER_DOCUMENT_REGISTRY, index=False)
    print(f"\n¡Registro Maestro creado con éxito en: {MASTER_DOCUMENT_REGISTRY}!")
    print("\nResumen actual del inventario:")
    print(df_registry[COL_ESTADO].value_counts())

if __name__ == "__main__":
    init_registry()