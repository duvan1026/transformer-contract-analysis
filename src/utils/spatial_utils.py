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


def build_annotation_registry(annotations: dict, contract_ids: list, valid_classes: list) -> pd.DataFrame:
    """
    Construye la verdad de referencia (documento x categoría) EXCLUSIVAMENTE
    a partir de las anotaciones originales de CUAD, para los contratos
    efectivamente evaluados en esta corrida de inferencia.

    No depende de qué fragmentos generó la fragmentación ni de qué predijo
    el modelo. Es la fuente correcta para el DENOMINADOR de Recall@K: una
    cláusula real que la fragmentación no logró capturar en ningún fragmento
    debe seguir contando en este registro (y por tanto, contar como fallo
    de recall), en lugar de desaparecer silenciosamente del cálculo.

    Incluye dos columnas de auditoría que no existen en ningún otro archivo
    del pipeline (no redundantes con ground_truth_results.csv ni batch_fragments.csv):
      - n_anotaciones_cuad: cuántos spans distintos de CUAD respaldan esa fila.
      - offsets_originales: offsets exactos "start-end" de cada span, separados
        por ';'.

    Args:
        annotations: dict devuelto por load_ground_truth() (doc_id en minúsculas -> lista de anotaciones).
        contract_ids: lista de id_documento efectivamente evaluados en esta corrida
                       (debe salir del registro maestro: estado="Utilizado" +
                       experimento de inferencia correspondiente, NO de los
                       fragmentos/predicciones generados).
        valid_classes: lista de categorías válidas (id2label.values()), sin normalizar.

    Returns:
        DataFrame con columnas ["id_documento", "categoria", "n_anotaciones_cuad",
        "offsets_originales"] (categoria en MAYÚSCULAS), una fila por cada
        combinación documento-categoría realmente anotada en CUAD.
    """
    valid_classes_lower = {str(c).strip().lower() for c in valid_classes}
    rows = []

    for doc_id in contract_ids:
        clean_id = str(doc_id).strip().lower()

        anns_by_cls = {}
        for ann in annotations.get(clean_id, []):
            if ann["label"] in valid_classes_lower:
                anns_by_cls.setdefault(ann["label"], []).append(ann)

        for cls, ann_list in anns_by_cls.items():
            offsets_str = ";".join(f'{a["start"]}-{a["end"]}' for a in ann_list)
            rows.append({
                "id_documento": doc_id,
                "categoria": cls.upper(),
                "n_anotaciones_cuad": len(ann_list),
                "offsets_originales": offsets_str,
            })

    return pd.DataFrame(
        rows,
        columns=["id_documento", "categoria", "n_anotaciones_cuad", "offsets_originales"]
    )


def _find_overlapping_matches(doc_id: str, frag_start: int, frag_end: int, annotations: dict) -> list:
    """
    Función interna compartida: calcula todas las anotaciones que solapan con
    el fragmento por encima del umbral, ordenadas de mayor a menor solape
    (y por offset de inicio como desempate).

    Cada elemento es una tupla (overlap_len, ann_start, ann_end, label) —
    se conserva ann_end además de ann_start para poder reconstruir el offset
    completo de la anotación ganadora en get_true_label_details().

    Único punto donde se aplica el criterio de desempate espacial:
    se selecciona la categoría con mayor solapamiento.
    """
    clean_doc_id = str(doc_id).strip().lower()
    if clean_doc_id not in annotations:
        return []

    matches = []
    for ann in annotations[clean_doc_id]:
        overlap_start = max(frag_start, ann["start"])
        overlap_end = min(frag_end, ann["end"])
        overlap_len = overlap_end - overlap_start
        if overlap_len >= OVERLAP_THRESHOLD_CHARS:
            matches.append((overlap_len, ann["start"], ann["end"], ann["label"]))

    matches.sort(key=lambda m: (-m[0], m[1]))
    return matches


def get_true_label_for_fragment(doc_id: str, frag_start: int, frag_end: int, annotations: dict) -> str:
    """
    Calcula la superposición espacial y devuelve SOLO la etiqueta real
    estandarizada en MAYÚSCULAS. No utiliza la predicción del modelo para calcular la etiqueta real,
    evitando circularidad entre predicción y ground truth.
    """
    matches = _find_overlapping_matches(doc_id, frag_start, frag_end, annotations)
    if not matches:
        return "NONE"
    return matches[0][3].upper()


def get_true_label_details(doc_id: str, frag_start: int, frag_end: int, annotations: dict) -> dict:
    """
    Igual que get_true_label_for_fragment(), pero devuelve además el detalle
    diagnóstico del cruce espacial, pensado para enriquecer
    ground_truth_results.csv y permitir auditoría directa sin recalcular nada:

      - n_anotaciones_solapadas: cuántos SPANS individuales de CUAD (sin
        deduplicar por categoría) competían por este fragmento. Puede ser
        mayor que n_categorias_distintas si varias cláusulas de la MISMA
        categoría caen dentro del mismo fragmento (señal de dilución por
        fragmento demasiado largo, no de ambigüedad entre clases).

      - n_categorias_distintas: cantidad de categorías ÚNICAS en competencia
        (= número de elementos en clases_solapadas). Es el indicador correcto
        para aislar ambigüedad inter-clase real separada de la dilución intra-clase (misma
        categoría repetida dentro de un fragmento demasiado largo).

      - overlap_chars_ganador: caracteres de solape de la cláusula ganadora.

      - offset_ganador: offset exacto "start-end" del span de CUAD que ganó
        el desempate. Permite recuperar el texto exacto de la cláusula
        decisiva sin volver a ejecutar load_ground_truth() ni reabrir
        CUAD_v1.json.

      - clases_solapadas: lista, separada por ';' y ordenada alfabéticamente,
        de TODAS las categorías únicas candidatas (no solo la ganadora).
        Agrupable directamente con value_counts() para detectar qué pares
        de categorías compiten con más frecuencia en el corpus.

    Esta es la función que debe usarse en 03_calculate_ground_truth.py.
    """
    matches = _find_overlapping_matches(doc_id, frag_start, frag_end, annotations)

    if not matches:
        return {
            "etiqueta_real": "NONE",
            "n_anotaciones_solapadas": 0,
            "n_categorias_distintas": 0,
            "overlap_chars_ganador": 0,
            "offset_ganador": "",
            "clases_solapadas": "",
        }

    clases = sorted({m[3].upper() for m in matches})
    ganador = matches[0]  # (overlap_len, ann_start, ann_end, label)

    return {
        "etiqueta_real": ganador[3].upper(),
        "n_anotaciones_solapadas": len(matches),
        "n_categorias_distintas": len(clases),
        "overlap_chars_ganador": ganador[0],
        "offset_ganador": f"{ganador[1]}-{ganador[2]}",
        "clases_solapadas": ";".join(clases),
    }