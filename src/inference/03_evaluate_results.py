"""
03_evaluate_results.py

Script orquestador de la Fase 3 (Evaluación del Pipeline y Simulación de Umbrales).
Realiza el cruce espacial pesado contra CUAD_v1 una sola vez. Luego, itera sobre 
una lista de umbrales configurables, recalculando las asignaciones finales 
y generando carpetas dinámicas con los reportes e imágenes para cada experimento.
"""

import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

# 1. Asegurar la inclusión de la raíz del proyecto en el PATH
sys.path.append(str(Path(__file__).resolve().parents[2]))

# Importar rutas globales
from src.config.paths import (
    INFERENCE_SCORED_RESULTS,
    INFERENCE_METRICS_REPORT,
    CUAD_JSON,
    CATEGORY_MAPPING_FILE,
    INFERENCE_CONFUSION_MATRIX_IMAGE
)

# Importación de estándares técnicos y flujos
from src.config.text_standards import (
    OVERLAP_THRESHOLD_CHARS,
    CONFIDENCE_THRESHOLDS
)

# Importar estándares y parámetros configurables
from src.config.label_mapping import ID2LABEL


def _load_category_mapping() -> dict:
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

def _load_ground_truth() -> dict:
    with open(CUAD_JSON, "r", encoding="utf-8") as f:
        cuad_data = json.load(f)["data"]
    category_map = _load_category_mapping()
    annotations = {}
    valid_classes = [label.lower().strip() for label in ID2LABEL.values()]
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

def _get_true_label_for_fragment(doc_id: str, frag_start: int, frag_end: int, pred_raw: str, annotations: dict) -> str:
    clean_doc_id = str(doc_id).strip().lower()
    if clean_doc_id not in annotations: return "NONE"
    matches = []
    for ann in annotations[clean_doc_id]:
        overlap_start = max(frag_start, ann["start"])
        overlap_end = min(frag_end, ann["end"])
        if (overlap_end - overlap_start) >= OVERLAP_THRESHOLD_CHARS:
            matches.append(ann["label"])
    if not matches: return "NONE"
    if pred_raw in matches: return pred_raw
    return matches[0]

def aplicar_umbral_dinamico(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    df_sim = df.copy()
    valid_classes = [label.lower().strip() for label in ID2LABEL.values()]
    predicciones = []
    
    for _, row in df_sim.iterrows():
        best_class, best_score = "NONE", -1.0
        for cls in valid_classes:
            var_under = f"score_{cls.replace(' ', '_').replace('-', '_')}"
            var_hyph = f"score_{cls.replace(' ', '_')}"
            col_name = var_hyph if var_hyph in df_sim.columns else var_under
            if col_name in df_sim.columns:
                score = float(row[col_name])
                if score > best_score:
                    best_score, best_class = score, cls
        predicciones.append(best_class if best_score >= threshold else "NONE")
    df_sim["pred_simulada"] = predicciones
    return df_sim

def evaluate_fragment_level(df: pd.DataFrame) -> str:
    y_true = df["etiqueta_real"]
    y_pred = df["pred_simulada"]  # Usamos la predicción recalculada por el umbral
    report = classification_report(y_true, y_pred, zero_division=0)
    output = "=== A. EVALUACIÓN A NIVEL FRAGMENTO (Classification Report) ===\n"
    output += "Perspectiva Técnica: Evalúa la precisión exacta del modelo por cada fragmento.\n"
    output += "• PRECISION: De lo que predijo como cláusula, cuánto era real.\n"
    output += "• RECALL: Del total de cláusulas reales, cuántas logró capturar.\n"
    output += "• F1-SCORE: Balance armónico entre Precisión y Cobertura.\n\n"
    output += report + "\n"
    output += "[NOTA] La Matriz de Confusión se ha guardado en un archivo PNG independiente.\n\n"
    return output

def evaluate_contract_level(df: pd.DataFrame) -> str:
    output = "=== B. EVALUACIÓN A NIVEL CONTRATO (Recall@K) ===\n"
    output += "Perspectiva Negocio: ¿La cláusula real está en el Top K de sugerencias del contrato?\n"
    output += "(Nota: Esta métrica es inmune al umbral, ya que evalúa el orden de probabilidad pura).\n\n"
    valid_classes = [label.lower() for label in ID2LABEL.values()]
    for cls in valid_classes:
        var_under = f"score_{cls.replace(' ', '_').replace('-', '_')}"
        var_hyph = f"score_{cls.replace(' ', '_')}"
        col_score = var_hyph if var_hyph in df.columns else var_under
            
        docs_with_class = df[df["etiqueta_real"] == cls]["id_documento"].unique()
        if len(docs_with_class) == 0: continue
            
        hits_at_1, hits_at_3 = 0, 0
        for doc in docs_with_class:
            doc_frags = df[df["id_documento"] == doc].copy()
            if col_score not in doc_frags.columns: continue
            top_frags = doc_frags.sort_values(by=col_score, ascending=False)
            top_1 = top_frags.head(1)["etiqueta_real"].values
            top_3 = top_frags.head(3)["etiqueta_real"].values
            if cls in top_1: hits_at_1 += 1
            if cls in top_3: hits_at_3 += 1
                
        r_at_1 = (hits_at_1 / len(docs_with_class)) * 100
        r_at_3 = (hits_at_3 / len(docs_with_class)) * 100
        output += f"Categoría: {cls.upper()} (Encontrada en {len(docs_with_class)} documentos)\n"
        output += f"  - Recall@1 : {r_at_1:.1f}%\n"
        output += f"  - Recall@3 : {r_at_3:.1f}%\n"
    return output + "\n"

def generate_mapping_summary() -> str:
    category_map = _load_category_mapping()
    if not category_map: return ""
    inverted_map = {}
    for orig_cat, final_cat in category_map.items():
        inverted_map.setdefault(final_cat, []).append(orig_cat)
    output = "=== C. CONTEXTO DE HOMOLOGACIÓN DINÁMICA (Excel) ===\n"
    output += "Consolidación de las cláusulas originales de CUAD v1:\n\n"
    for final_label, original_list in sorted(inverted_map.items()):
        output += f"• Categoría Unificada: {final_label.upper()}\n"
        for orig in sorted(original_list):
            output += f"    - {orig}\n"
        output += "\n"
    return output

def save_confusion_matrix_image(df: pd.DataFrame, output_path: Path, umbral: float):
    labels_order = ["NONE", "audit rights", "governing law", "limitation of liability", "non-compete", "termination for convenience"]
    cm = confusion_matrix(df["etiqueta_real"], df["pred_simulada"], labels=labels_order)
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="white")
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[lbl.upper() for lbl in labels_order], yticklabels=[lbl.upper() for lbl in labels_order], square=True, annot_kws={"size": 11, "weight": "bold"})
    plt.title(f"Matriz de Confusión - Umbral {int(umbral*100)}%", fontsize=14, pad=20, weight='bold')
    plt.xlabel("Predicción del Modelo", fontsize=12, labelpad=15, weight='bold')
    plt.ylabel("Etiqueta Real", fontsize=12, labelpad=15, weight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("[*] Iniciando Evaluación Base (Cálculo Espacial)...")
    df_results = pd.read_csv(INFERENCE_SCORED_RESULTS)
    annotations = _load_ground_truth()
    
    # 1. Trabajo Pesado (Se hace UNA SOLA VEZ)
    df_results["etiqueta_real"] = df_results.apply(
        lambda row: _get_true_label_for_fragment(row["id_documento"], row["start_offset"], row["end_offset"], row["pred_raw"], annotations), axis=1
    )
    
    # Actualizar el archivo maestro con la nueva columna
    df_results.to_csv(INFERENCE_SCORED_RESULTS, index=False, encoding="utf-8")
    print(f"[+] Archivo maestro actualizado con 'etiqueta_real'.")
    
    # Textos comunes
    report_C = generate_mapping_summary()
    
    # 2. Bucle de Simulación de Umbrales
    print(f"[*] Iniciando simulación para los umbrales: {CONFIDENCE_THRESHOLDS}")
    
    for umbral in CONFIDENCE_THRESHOLDS:
        print(f"\n--- Procesando Umbral {umbral} ---")
        
        # A. Asignar clases dinámicamente según el umbral actual
        df_sim = aplicar_umbral_dinamico(df_results, umbral)
        
        # B. Calcular reportes
        report_A = evaluate_fragment_level(df_sim)
        report_B = evaluate_contract_level(df_sim)
        final_report = f"REPORTE DE EVALUACIÓN - UMBRAL: {umbral}\n\n" + report_A + report_B + report_C
        
        # C. Crear la subcarpeta dinámica
        nombre_carpeta = f"threshold_{int(umbral * 100)}"
        carpeta_experimento = INFERENCE_METRICS_REPORT.parent / nombre_carpeta
        carpeta_experimento.mkdir(parents=True, exist_ok=True)
        
        # D. Rutas de guardado
        ruta_reporte = carpeta_experimento / INFERENCE_METRICS_REPORT.name
        ruta_imagen = carpeta_experimento / INFERENCE_CONFUSION_MATRIX_IMAGE.name
        
        # E. Guardar archivos
        with open(ruta_reporte, "w", encoding="utf-8") as f:
            f.write(final_report)
            
        save_confusion_matrix_image(df_sim, ruta_imagen, umbral)
        print(f"  [+] Guardado en: {carpeta_experimento.name}/")

    print("\n[*] FASE 3 Y SIMULACIÓN COMPLETADAS CON ÉXITO.")

if __name__ == "__main__":
    main()