"""
src/utils/spatial_utils.py

Herramientas especializadas para el cálculo de superposiciones espaciales 
y asignación de la verdad terreno (Ground Truth) basándose en las 
anotaciones originales de los abogados en CUAD_v1.json.
"""

import json
import pandas as pd
from src.config.paths import CUAD_JSON, CATEGORY_MAPPING_FILE
from src.config.text_standards import OVERLAP_THRESHOLD_CHARS

def load_category_mapping() -> dict:
    """Carga y limpia el mapeo de categorías desde el Excel oficial."""
    if not CATEGORY_MAPPING_FILE.exists():
        return {}
        
    df_map = pd.read_excel(CATEGORY_MAPPING_FILE)
    df_map.columns = df_map.columns.str.strip()
    mapping_dict = {}
    required_cols = ['Category', 'Usar', 'Final Name Category']
    
    if all(col in df_map.columns for col in required_cols):
        df_map['Usar_clean'] = df_map['Usar'].astype(str).str.strip().str.lower()
        df_usar = df_map[df_map['Usar_clean'].isin(['sí', 'si', 'yes'])]
        
        for _, row in df_usar.iterrows():
            orig_cat = str(row['Category']).strip().lower()
            final_cat = str(row['Final Name Category']).strip().lower()
            if final_cat and final_cat != 'nan' and final_cat != '-':
                mapping_dict[orig_cat] = final_cat
                
    return mapping_dict


def load_ground_truth(id2label: dict) -> dict:
    """Carga el JSON de CUAD y extrae las coordenadas reales por documento."""
    with open(CUAD_JSON, "r", encoding="utf-8") as f:
        cuad_data = json.load(f)["data"]
        
    category_map = load_category_mapping()
    annotations = {}
    valid_classes = [label.lower().strip() for label in id2label.values()]
    
    for doc in cuad_data:
        doc_title = doc["title"].strip().lower()
        annotations[doc_title] = []
        for paragraph in doc["paragraphs"]:
            for qa in paragraph["qas"]:
                if not qa["is_impossible"]:
                    raw_cuad_label = qa["id"].split("__")[1].replace("_", " ").strip().lower()
                    mapped_label = category_map.get(raw_cuad_label, raw_cuad_label)
                    
                    if mapped_label in valid_classes:
                        for answer in qa["answers"]:
                            annotations[doc_title].append({
                                "label": mapped_label,
                                "start": answer["answer_start"],
                                "end": answer["answer_start"] + len(answer["text"])
                            })
                            
    return annotations


def get_true_label_for_fragment(doc_id: str, frag_start: int, frag_end: int, pred_raw: str, annotations: dict) -> str:
    """Calcula la superposición espacial y devuelve la etiqueta real estandarizada en MAYÚSCULAS."""
    clean_doc_id = str(doc_id).strip().lower()
    
    # Si el documento no está en las anotaciones, es ruido/NONE
    if clean_doc_id not in annotations: 
        return "NONE"
        
    matches = []
    
    # Revisar contra todas las anotaciones de ese documento específico
    for ann in annotations[clean_doc_id]:
        overlap_start = max(frag_start, ann["start"])
        overlap_end = min(frag_end, ann["end"])
        
        # Validación matemática del umbral de superposición
        if (overlap_end - overlap_start) >= OVERLAP_THRESHOLD_CHARS:
            matches.append(ann["label"])
            
    if not matches: 
        return "NONE"
        
    pred_lower = str(pred_raw).strip().lower()
    final_label = pred_lower if pred_lower in matches else matches[0]
    
    return final_label.upper()