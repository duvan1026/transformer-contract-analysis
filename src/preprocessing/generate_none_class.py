import sys
from pathlib import Path
import json
import random
import pandas as pd
import numpy as np
import re

# Añadir raíz del proyecto al path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config.paths import (
    CUAD_JSON,
    DATASET_TAXONOMY_CSV,
    DATASET_WITH_NONE_CSV
)

# Importamos las constantes Estandarizadas
from src.config.text_standards import (
    RANDOM_SEED,
    MIN_CHARS_THRESHOLD,
    FILTER_PURE_NUMBERS,
    NUMERIC_CLEAN_REGEX,
    SENTENCE_BOUNDARY_REGEX
)


def set_seed(seed: int):
    """Garantiza el determinismo en la selección aleatoria."""
    random.seed(seed)
    np.random.seed(seed)


def load_contracts_context(json_path: Path) -> dict:
    """
    Lee el CUAD.json original y extrae el texto completo (context) de cada contrato.
    Retorna un diccionario: { 'id_documento': 'texto completo...' }
    """
    contexts = {}
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for document in data["data"]:
        for paragraph in document["paragraphs"]:
            if paragraph.get("qas"):
                first_qa_id = paragraph["qas"][0]["id"]
                id_documento = first_qa_id.split("__")[0]
                contexts[id_documento] = paragraph["context"]
                break  
                
    return contexts


def build_forbidden_zones(df_taxonomy: pd.DataFrame) -> dict:
    """
    Procesa el CSV de categorías válidas y genera un mapa de intervalos ocupados por contrato.
    Retorna: { 'id_documento': [(inicio_1, fin_1), (inicio_2, fin_2)] }
    """
    forbidden_zones = {}
    for _, row in df_taxonomy.iterrows():
        doc_id = row["id_documento"]
        start = int(row["offset"])
        text_length = len(str(row["span_text"]))
        end = start + text_length

        if doc_id not in forbidden_zones:
            forbidden_zones[doc_id] = []
        forbidden_zones[doc_id].append((start, end))
        
    return forbidden_zones


def extract_sentences_with_offsets(text: str) -> list:
    """
    Segmenta un texto utilizando la expresión regular oficial del proyecto,
    devolviendo los índices exactos (offsets) para garantizar trazabilidad.
    """
    sentences = []

    # re.finditer escanea el texto y aplica regla oficial de corte
    for match in re.finditer(SENTENCE_BOUNDARY_REGEX, text):
        s_text = match.group(0) # El texto capturado
        s_start = match.start() # Índice de inicio
        s_end = match.end()     # Índice de fin
        
        # Limpiar espacios vacíos a los lados sin perder los índices originales
        if s_text.strip():
            sentences.append({
                "text": s_text.strip(),
                "start": s_start,
                "end": s_end
            })
            
    return sentences


def extract_none_for_document(doc_id: str, text: str, zones: list, n_needed: int) -> list:
    """
    Extrae de forma reproducible N oraciones 'none' de un documento específico,
    aplicando estrictamente las reglas de auditoría de text_standards.py.
    """
    sentences = extract_sentences_with_offsets(text)
    valid_candidates = []

    for sentence in sentences:
        s_text = sentence["text"]
        s_start = sentence["start"]
        s_end = sentence["end"]
        
        # Filtro de Calidad por Longitud Mínima Oficial
        if len(s_text) < MIN_CHARS_THRESHOLD:
            continue

        # Filtro de Fragmentos Puramente Numéricos o Ruido
        if FILTER_PURE_NUMBERS:
            # Usamos tu regex oficial para remover puntos, guiones, barras y espacios
            cleaned_text = re.sub(NUMERIC_CLEAN_REGEX, '', s_text)
            
            # Si después de limpiar el texto, lo que queda son solo dígitos o está vacío,
            # significa que era una numeración (ej: "1.1.2" -> "112") o puros símbolos.
            if cleaned_text.isdigit() or cleaned_text == "":
                continue

        # Verificar colisión matemática contra las zonas ocupadas de ESTE contrato
        collision = False
        for z_start, z_end in zones:
            if (s_start < z_end) and (s_end > z_start):
                collision = True
                break

        if not collision:
            valid_candidates.append({
                "id_documento": doc_id,
                "categoria": "none",
                "span_text": sentence["text"],
                "offset": s_start
            })

    # Muestreo aleatorio local controlado por la semilla
    if len(valid_candidates) < n_needed:
        # Si el contrato es muy corto y no hay suficientes oraciones libres, tomamos las que haya
        return valid_candidates
    
    return random.sample(valid_candidates, n_needed)


def run():
    print("Iniciando generación de clase 'none' con BALANCE LOCAL POR CONTRATO...")
    
    # 1. Fijar semilla global para asegurar reproducibilidad exacta
    set_seed(RANDOM_SEED)

    # 2. Cargar dataset de taxonomía homologada
    df_taxonomy = pd.read_csv(DATASET_TAXONOMY_CSV)
    
    # 3. Cargar contextos de contratos y construir mapa de zonas prohibidas
    contexts = load_contracts_context(CUAD_JSON)
    forbidden_zones = build_forbidden_zones(df_taxonomy)

    all_sampled_nones = []
    
    # 4. Agrupar la taxonomía por contrato para calcular el balance local
    grouped = df_taxonomy.groupby("id_documento")
    
    print(f"Procesando {len(grouped)} contratos únicos para balancear...")

    for doc_id, group in grouped:
        # Calcular el número máximo de spans que tiene la clase más frecuente EN ESTE CONTRATO
        conteo_clases_local = group["categoria"].value_counts()
        n_max_local = conteo_clases_local.max()
        
        # Obtener el texto del contrato y sus zonas prohibidas
        text = contexts.get(doc_id)
        if not text:
            print(f"⚠️ Advertencia: No se encontró el texto original para el contrato {doc_id}")
            continue
            
        zones = forbidden_zones.get(doc_id, [])
        
        # Extraer exactamente n_max_local muestras para este contrato
        doc_nones = extract_none_for_document(doc_id, text, zones, n_max_local)
        all_sampled_nones.extend(doc_nones)

    # 5. Consolidar e integrar datasets
    df_nones = pd.DataFrame(all_sampled_nones)
    print(f"Se generaron exitosamente {len(df_nones)} spans para la clase 'none' en total.")

    df_final = pd.concat([df_taxonomy, df_nones], ignore_index=True)
    
    # Ordenar por documento y offset para mantener coherencia de lectura
    df_final = df_final.sort_values(by=["id_documento", "offset"]).reset_index(drop=True)

    # 6. Guardar el nuevo dataset extendido
    Path(DATASET_WITH_NONE_CSV).parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(DATASET_WITH_NONE_CSV, index=False)
    
    print(f"✅ Éxito! Dataset balanceado guardado en: {DATASET_WITH_NONE_CSV}")
    print("\nDistribución final de clases general:")
    print(df_final["categoria"].value_counts())


if __name__ == "__main__":
    run()