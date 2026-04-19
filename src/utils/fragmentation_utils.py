import re

def fragment_single_document(document_id, raw_text, tokenizer, max_tokens=512):
    """
    Estrategia Relacional Determinista:
    - id_documento: Nombre del contrato.
    - parrafo_id: ID numérico del párrafo original (para trazabilidad de hijos).
    - fragmento_id: ID numérico secuencial de todos los trozos del documento.
    """
    final_fragments = []
    f_idx = 0  # Contador global de fragmentos para este documento
    p_idx = 0  # Contador de párrafos originales
    
    # 1. Separación primaria: Buscar bloques de texto separados por líneas en blanco (\n\n)
    # Esta expresión regular atrapa párrafos completos respetando sus offsets
    for match in re.finditer(r'([^\n]+(?:\n[^\n]+)*)', raw_text):
        para_text = match.group(1).strip()
        if not para_text:
            continue
            
        p_idx += 1  # Nuevo párrafo encontrado
        para_start = match.start(1)
        
        # Medimos el párrafo con el tokenizador
        tokens = tokenizer.encode(para_text, add_special_tokens=False)
        
        # 2. Si el párrafo cabe perfectamente, es un fragmento válido
        if len(tokens) <= max_tokens:
            f_idx += 1
            final_fragments.append({
                "id_documento": document_id,
                "parrafo_id": p_idx,
                "fragmento_id": f_idx,
                "start_offset": para_start,
                "end_offset": para_start + len(para_text),
                "contenido": para_text,
                "tokens_estimados": len(tokens)
            })
        else:
            # 3. Caso: Párrafo gigante que requiere subdivisión determinista
            # Esta función ahora devuelve una lista de tuplas (texto, start, end)
            sub_chunks = _split_long_paragraph_deterministically(
                text=para_text, 
                start_offset=para_start, 
                tokenizer=tokenizer, 
                max_tokens=max_tokens
            )

            for sub_text, s_start, s_end in sub_chunks:
                f_idx += 1
                final_fragments.append({
                    "id_documento": document_id,
                    "parrafo_id": p_idx,  # Todos los hijos comparten el mismo ID de su padre
                    "fragmento_id": f_idx,
                    "start_offset": s_start,
                    "end_offset": s_end,
                    "contenido": sub_text,
                    "tokens_estimados": len(tokenizer.encode(sub_text, add_special_tokens=False))
                })
    
    return final_fragments

def _split_long_paragraph_deterministically(text, start_offset, tokenizer, max_tokens):
    """
    Divide el párrafo acumulando oraciones hasta alcanzar el límite.
    Retorna lista de tuplas: [(texto, start, end), ...]
    """
    chunks = []
    current_text = ""
    current_start = -1
    
    # Dividir por oraciones (buscando puntos, signos o saltos de línea)
    for match in re.finditer(r'[^.!?\n]+[.!?\n]*', text):
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