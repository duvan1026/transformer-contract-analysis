"""
generate_none_class.py

Script de extracción y consolidación de la clase NONE (Activo Maestro).
Procesa el universo completo de contratos válidos (454) aplicando un pipeline
de "chunking" (fragmentación por párrafos), filtrado por longitud, cálculo de 
densidad alfabética y exclusión de zonas de colisión espacial.

El resultado se consolida en un Documento Maestro, del cual posteriormente
se extraerán las muestras para entrenamiento.
"""

import sys
import re
import json
import random
import pandas as pd
import numpy as np
from pathlib import Path


# Añadir raíz del proyecto al path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config.paths import (
    CUAD_JSON,
    DATASET_TAXONOMY_CSV,
    DATASET_NONE_MASTER_CSV
)

# Importamos las constantes Estandarizadas
from src.config.text_standards import (
    RANDOM_SEED,
    MIN_CHARS_THRESHOLD,
    NONE_MIN_CHARS_THRESHOLD,
    NONE_MIN_ALPHA_DENSITY,
    PARAGRAPH_DELIMITER,
    ALPHABETIC_CHARS_REGEX,
)

from src.utils.text_utils import clean_text


def set_seed(seed: int):
    """Garantiza el determinismo en la selección aleatoria."""
    random.seed(seed)
    np.random.seed(seed)


def load_contracts_context(json_path: Path) -> dict:
    """
    Lee el CUAD.json original y extrae el texto completo de cada contrato.
    Retorna: Diccionario { 'id_documento': 'texto completo...' }
    """
    contexts = {}
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for document in data["data"]:
        doc_name = document["title"]
        # Extraer el texto completo del documento (context)
        context = document["paragraphs"][0]["context"]
        contexts[doc_name] = context
        
    return contexts


def build_forbidden_zones(df_taxonomy: pd.DataFrame) -> dict:
    """
    Fase 1: Mapeo de Zonas Prohibidas (Seguridad Espacial).
    Extrae las coordenadas exactas donde existen cláusulas positivas.
    Retorna: Diccionario { 'id_documento': [(start_offset, end_offset), ...] }
    """
    forbidden_zones = {}
    for _, row in df_taxonomy.iterrows():
        doc_id = row["id_documento"]
        # El inicio es el offset guardado
        start = int(row["offset"])
        # El fin se calcula con la longitud exacta del texto guardado
        end = start + len(str(row["span_text"]))
        
        if doc_id not in forbidden_zones:
            forbidden_zones[doc_id] = []
        forbidden_zones[doc_id].append((start, end))
        
    return forbidden_zones


def calculate_alpha_density(text: str) -> float:
    """
    Fase 4 (Subrutina): Calcula el porcentaje de letras en un texto.
    """
    if not text:
        return 0.0
    # Encontrar todas las letras puras
    alpha_chars = re.findall(ALPHABETIC_CHARS_REGEX, text)
    return len(alpha_chars) / len(text)


def process_single_document(doc_id: str, text: str, zones: list) -> list:
    """
    Fases 2 a 5: Aplica el embudo de filtrado a un único contrato.
    Retorna una lista de diccionarios con los fragmentos NONE validados.
    """
    valid_chunks = []
    
    # Fase 2: Fragmentación Estructural (Chunking) por Párrafos
    for match in re.finditer(PARAGRAPH_DELIMITER, text):
        chunk_text = match.group(0)
        start_offset = match.start()
        end_offset = match.end()

        # Variable para evalaur texto (Sin espacios)
        eval_text = clean_text(chunk_text)
        
        # Fase 3: Auditoría Física (Filtro de Longitud para evitar atajo)
        if len(eval_text) < NONE_MIN_CHARS_THRESHOLD:
            continue
            
        # Fase 4: Auditoría Lógica (Densidad Alfabética)
        density = calculate_alpha_density(eval_text)
        if density < NONE_MIN_ALPHA_DENSITY:
            continue
            
        # Fase 5: Auditoría Espacial (Filtro de Colisión)
        collision = False
        for z_start, z_end in zones:
            # Fórmula de intersección matemática: Inicio A < Fin B Y Fin A > Inicio B
            if (start_offset < z_end) and (end_offset > z_start):
                collision = True
                break
                
        # Si sobrevive a todas las auditorías, es un candidato NONE perfecto
        if not collision:
            valid_chunks.append({
                "id_documento": doc_id,
                "categoria": "NONE",
                "span_text": chunk_text,
                "start_offset": start_offset,
                "end_offset": end_offset,
                "alphabetic_density": round(density, 4)
            })
            
    return valid_chunks


def main():
    print(f"\n[INFO] Iniciando pipeline de extracción Master NONE...")
    set_seed(RANDOM_SEED)

    # 1. Cargar el texto fuente y las taxonomías positivas
    print("[INFO] Cargando contextos crudos desde CUAD_v1.json...")
    contexts = load_contracts_context(CUAD_JSON)
    
    print(f"[INFO] Cargando archivo de taxonomía: {DATASET_TAXONOMY_CSV.name}...")
    try:
        df_taxonomy = pd.read_csv(DATASET_TAXONOMY_CSV)
    except FileNotFoundError:
        print(f"[ERROR] No se encontró {DATASET_TAXONOMY_CSV}. Debe generarse primero.")
        sys.exit(1)

    # 2. Construir zonas prohibidas
    print("[INFO] Mapeando zonas espaciales prohibidas (cláusulas reales)...")
    forbidden_zones = build_forbidden_zones(df_taxonomy)
    
    # 3. Procesar el embudo documento por documento
    all_none_master_records = []
    doc_ids_to_process = df_taxonomy["id_documento"].unique()
    
    print(f"[INFO] Procesando chunking y filtros para {len(doc_ids_to_process)} documentos...")
    
    for doc_id in doc_ids_to_process:
        text = contexts.get(doc_id)
        if not text:
            print(f"[WARN] No se encontró texto original para {doc_id}. Omitiendo.")
            continue
            
        zones = forbidden_zones.get(doc_id, [])
        doc_valid_nones = process_single_document(doc_id, text, zones)
        all_none_master_records.extend(doc_valid_nones)

    # Fase 6: Consolidación en el Master File
    df_master_none = pd.DataFrame(all_none_master_records)
    
    print("\n" + "="*50)
    print("               RESUMEN DE EXTRACCIÓN NONE")
    print("="*50)
    print(f"Total Documentos Procesados: {len(doc_ids_to_process)}")
    print(f"Total Fragmentos Master NONE Extraídos: {len(df_master_none)}")
    
    if not df_master_none.empty:
        # Asegurar que el directorio de salida exista
        DATASET_NONE_MASTER_CSV.parent.mkdir(parents=True, exist_ok=True)
        
        # Guardar en disco
        df_master_none.to_csv(DATASET_NONE_MASTER_CSV, index=False, encoding="utf-8")
        print(f"\n[ÉXITO] Archivo maestro guardado correctamente en: {DATASET_NONE_MASTER_CSV}")
    else:
        print("\n[ERROR] No se extrajo ningún fragmento válido. Revise los umbrales.")

if __name__ == "__main__":
    main()