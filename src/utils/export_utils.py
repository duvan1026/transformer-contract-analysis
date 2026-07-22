"""
src/utils/export_utils.py

Módulo dedicado exclusivamente a la generación de archivos de salida estructurados (JSON).
Separa la lógica de I/O de la lógica matemática o visual.
"""

import json
import pandas as pd
import datetime
from pathlib import Path

def generate_contract_jsons(df: pd.DataFrame, umbral: float, carpeta_experimento: Path, id2label: dict):
    """
    Agrupa los fragmentos predichos por contrato y genera un archivo JSON por cada uno.
    Estructura la salida para que sea consumible por un sistema o interfaz externa.
    
    Args:
        df: DataFrame con las predicciones finales.
        umbral: El umbral aplicado (para registro de trazabilidad).
        carpeta_experimento: Directorio donde se creará la carpeta 'json_outputs'.
    """
    json_dir = carpeta_experimento / "json_outputs"
    json_dir.mkdir(parents=True, exist_ok=True)
    
    valid_classes = [label.lower().strip() for label in id2label.values()]
    processing_date = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    
    # Tolerancia por si la columna de texto cambia de nombre en futuras versiones
    texto_col = 'contenido' if 'contenido' in df.columns else 'text'
    
    contratos = df.groupby("id_documento")
    jsons_creados = 0
    
    for id_doc, doc_df in contratos:
        doc_data = {
            "document_id": str(id_doc),
            "processing_date": processing_date,
            "applied_threshold": float(umbral),
            "extracted_clauses": {cls: [] for cls in valid_classes}
        }
        
        # Filtramos la "basura" (NONE) para no incluirla en el JSON final
        valid_frags = doc_df[doc_df["pred_final"] != "NONE"]
        
        for cls in valid_classes:
            frags_cls = valid_frags[valid_frags["pred_final"] == cls.upper()]
            if frags_cls.empty:
                continue
            
            # Buscamos la columna de score correspondiente a esta clase
            var_under = f"score_{cls.replace(' ', '_').replace('-', '_')}"
            col_score = var_under if var_under in df.columns else f"score_{cls.replace(' ', '_')}"
            
            # Ordenamos por las predicciones con mayor probabilidad matemática
            if col_score in frags_cls.columns:
                frags_cls = frags_cls.sort_values(by=col_score, ascending=False)
            
            # Construimos el diccionario de la cláusula
            for _, row in frags_cls.iterrows():
                score_val = float(row[col_score]) if col_score in row else float(row.get("confidence_raw", 0.0))
                start_off = row.get("start_offset")
                end_off = row.get("end_offset")
                
                doc_data["extracted_clauses"][cls].append({
                    "score": round(score_val, 4),
                    "text_snippet": str(row.get(texto_col, "")),
                    "start_offset": int(start_off) if pd.notna(start_off) else None,
                    "end_offset": int(end_off) if pd.notna(end_off) else None
                })
                
        # Guardado físico del JSON usando un nombre seguro
        safe_doc_id = str(id_doc).replace('.txt', '').replace('/', '_')
        out_file = json_dir / f"{safe_doc_id}.json"
        
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(doc_data, f, indent=4, ensure_ascii=False)
            
        jsons_creados += 1
        
    print(f"  [+] Generados {jsons_creados} archivos JSON en: {json_dir.name}/")