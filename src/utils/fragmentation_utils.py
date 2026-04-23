import re
from src.utils.quality_filters import FragmentQualityFilter

from src.config.text_standards import (
    PARAGRAPH_DELIMITER,
    SENTENCE_BOUNDARY_REGEX,
    MAX_TOKENS
)

def fragment_single_document(document_id, raw_text, tokenizer, max_tokens=MAX_TOKENS):
    """
    Estrategia Relacional Determinista:
    - id_documento: Nombre del contrato.
    - parrafo_id: ID numérico del párrafo original (para trazabilidad de hijos).
    - fragmento_id: ID numérico secuencial de todos los trozos del documento.
    """
    # final_fragments = []
    valid_fragments = []
    rejected_fragments = []

    f_idx = 0  # Contador global de fragmentos para este documento
    p_idx = 0  # Contador de párrafos originales
    
    # 1. Separación primaria: Buscar bloques de texto separados por líneas en blanco (\n\n)
    # Esta expresión regular atrapa párrafos completos respetando sus offsets
    for match in re.finditer(PARAGRAPH_DELIMITER, raw_text):
        para_text = match.group(1).strip()
        if not para_text:
            continue
            
        para_start = match.start(1)
        para_end = match.end(1)
        
        # Medimos el párrafo con el tokenizador
        tokens = tokenizer.encode(para_text, add_special_tokens=False)
        
        # --- CASO A: El párrafo cabe entero ---
        if len(tokens) <= max_tokens:
            is_valido, motivo = FragmentQualityFilter.validate(para_text)
            
            if is_valido:
                p_idx += 1 # Solo aumenta si el párrafo es útil
                f_idx += 1
                valid_fragments.append({
                    "id_documento": document_id,
                    "parrafo_id": p_idx,
                    "fragmento_id": f_idx,
                    "start_offset": para_start,
                    "end_offset": para_end,
                    "contenido": para_text,
                    "tokens_estimados": len(tokens)
                })
            else:
                # Se va a la auditoría, NO aumenta contadores
                rejected_fragments.append({
                    "id_documento": document_id,
                    "start_offset": para_start,
                    "end_offset": para_end,
                    "contenido": para_text,
                    "motivo_descarte": motivo
                })
        
        # --- CASO B: Párrafo gigante (Subdivisión) ---
        else:
            sub_chunks = _split_long_paragraph_deterministically(
                para_text, para_start, tokenizer, max_tokens
            )
            
            # Filtramos los sub-chunks antes de asignarles IDs
            valid_subs = []
            for s_text, s_start, s_end in sub_chunks:
                is_valido, motivo = FragmentQualityFilter.validate(s_text)
                if is_valido:
                    valid_subs.append((s_text, s_start, s_end))
                else:
                    rejected_fragments.append({
                        "id_documento": document_id,
                        "start_offset": s_start,
                        "end_offset": s_end,
                        "contenido": s_text,
                        "motivo_descarte": motivo
                    })
            
            # Si después de filtrar quedaron hijos válidos, los guardamos bajo un mismo p_idx
            if valid_subs:
                p_idx += 1
                for s_text, s_start, s_end in valid_subs:
                    f_idx += 1
                    t_count = len(tokenizer.encode(s_text, add_special_tokens=False))
                    valid_fragments.append({
                        "id_documento": document_id,
                        "parrafo_id": p_idx,
                        "fragmento_id": f_idx,
                        "start_offset": s_start,
                        "end_offset": s_end,
                        "contenido": s_text,
                        "tokens_estimados": t_count
                    })
    
    # Devolvemos la tupla de dos listas
    return valid_fragments, rejected_fragments

def _split_long_paragraph_deterministically(text, start_offset, tokenizer, max_tokens):
    """
    Divide el párrafo acumulando oraciones hasta alcanzar el límite.
    Retorna lista de tuplas: [(texto, start, end), ...]
    """
    chunks = []
    current_text = ""
    current_start = -1
    
    # Dividir por oraciones (buscando puntos, signos o saltos de línea)
    for match in re.finditer(SENTENCE_BOUNDARY_REGEX, text):
        sentence_text = match.group(0)
        sentence_abs_start = start_offset + match.start()
        
        if current_start == -1:
            current_start = sentence_abs_start
            
        test_text = current_text + sentence_text
        test_tokens = len(tokenizer.encode(test_text, add_special_tokens=False))
        
        if test_tokens <= max_tokens:
            current_text = test_text
        else:
            # Cerramos el chunk actual antes de que desborde
            if current_text.strip():
                chunks.append((
                    current_text.strip(), 
                    current_start, 
                    current_start + len(current_text)
                ))
            
            # Iniciamos el siguiente chunk con la oración actual
            current_text = sentence_text
            current_start = sentence_abs_start
            
    # Guardar el residuo final
    if current_text.strip():
        chunks.append((
            current_text.strip(), 
            current_start, 
            current_start + len(current_text)
        ))
        
    return chunks