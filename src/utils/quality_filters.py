import re
from src.config.text_standards import (
    PARAGRAPH_DELIMITER,
    NUMERIC_CLEAN_REGEX,
    FILTER_PURE_NUMBERS,
    SENTENCE_BOUNDARY_REGEX
)

class FragmentQualityFilter:
    """Encapsula las reglas de negocio para validar la calidad de un fragmento."""
    
    @staticmethod
    def validate(text, min_length=20):
        """
        Retorna (es_valido, motivo)
        """
        text_strip = text.strip()
        
        # 1. Regla de Longitud mínima
        if len(text_strip) < min_length:
            return False, f"Longitud insuficiente (< {min_length} caracteres)"
        
        # 2. Regla de Contenido Numérico
        if FILTER_PURE_NUMBERS:
            # Eliminamos puntos, comas, guiones y espacios para ver si solo quedan números
            only_numbers = re.sub(NUMERIC_CLEAN_REGEX, '', text_strip)
            if only_numbers.isdigit() and len(only_numbers) > 0:
                return False, "Contenido puramente numérico"
        
        return True, None

def fragment_single_document(document_id, raw_text, tokenizer, max_tokens=512):
    """
    Fragmenta y clasifica los trozos en Válidos o Rechazados (Auditoría).
    """
    valid_fragments = []
    rejected_fragments = []
    
    f_idx = 0  # Contador global de fragmentos VÁLIDOS
    p_idx = 0  # Contador de párrafos originales VÁLIDOS

    # Separación por párrafos naturales
    for match in re.finditer(PARAGRAPH_DELIMITER, raw_text):
        para_text = match.group(1).strip()
        if not para_text:
            continue
            
        para_start = match.start(1)
        para_end = match.end(1)
        
        # Medir tokens
        tokens = tokenizer.encode(para_text, add_special_tokens=False)
        
        # Caso A: Párrafo corto (cabe en BERT)
        if len(tokens) <= max_tokens:
            is_valido, motivo = FragmentQualityFilter.validate(para_text)
            
            if is_valido:
                p_idx += 1
                f_idx += 1
                valid_fragments.append(_build_dict(document_id, p_idx, f_idx, para_start, para_end, para_text, len(tokens)))
            else:
                rejected_fragments.append(_build_rejected_dict(document_id, para_start, para_end, para_text, motivo))
        
        # Caso B: Párrafo largo (requiere subdivisión)
        else:
            sub_chunks = _split_long_paragraph_deterministically(para_text, para_start, tokenizer, max_tokens)
            
            # El parrafo_id solo aumenta si al menos un hijo es válido
            temp_valid_subchunks = []
            
            for sub_text, s_start, s_end in sub_chunks:
                is_valido, motivo = FragmentQualityFilter.validate(sub_text)
                
                if is_valido:
                    temp_valid_subchunks.append((sub_text, s_start, s_end))
                else:
                    rejected_fragments.append(_build_rejected_dict(document_id, s_start, s_end, sub_text, motivo))
            
            # Si hubo hijos válidos, procesarlos bajo un mismo parrafo_id
            if temp_valid_subchunks:
                p_idx += 1
                for sub_text, s_start, s_end in temp_valid_subchunks:
                    f_idx += 1
                    t_count = len(tokenizer.encode(sub_text, add_special_tokens=False))
                    valid_fragments.append(_build_dict(document_id, p_idx, f_idx, s_start, s_end, sub_text, t_count))

    return valid_fragments, rejected_fragments

def _build_dict(doc_id, p_id, f_id, start, end, text, tokens):
    return {
        "id_documento": doc_id,
        "parrafo_id": p_id,
        "fragmento_id": f_id,
        "start_offset": start,
        "end_offset": end,
        "contenido": text,
        "tokens_estimados": tokens
    }

def _build_rejected_dict(doc_id, start, end, text, reason):
    return {
        "id_documento": doc_id,
        "start_offset": start,
        "end_offset": end,
        "contenido": text,
        "motivo_descarte": reason
    }

def _split_long_paragraph_deterministically(text, start_offset, tokenizer, max_tokens):
    """Lógica de división por oraciones (determinista)"""
    chunks = []
    current_text = ""
    current_start = -1
    
    for match in re.finditer(SENTENCE_BOUNDARY_REGEX, text):
        sentence_text = match.group(0)
        sentence_abs_start = start_offset + match.start()
        
        if current_start == -1: current_start = sentence_abs_start
            
        test_tokens = len(tokenizer.encode(current_text + sentence_text, add_special_tokens=False))
        
        if test_tokens <= max_tokens:
            current_text += sentence_text
        else:
            if current_text.strip():
                chunks.append((current_text.strip(), current_start, current_start + len(current_text)))
            current_text, current_start = sentence_text, sentence_abs_start
            
    if current_text.strip():
        chunks.append((current_text.strip(), current_start, current_start + len(current_text)))
    return chunks