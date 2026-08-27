"""
src/utils/rebalancing/splitting.py

Núcleo puro del rebalanceo de unidades de texto largas (spans de train o
fragmentos de inferencia). No depende de pandas ni de rutas de disco: opera
sobre (texto, offset_absoluto_de_inicio, tokenizer, umbral) y devuelve una
lista de piezas con sus offsets absolutos recalculados.

Estrategia (cascada de niveles, confirmada con datos reales de
ground_truth_results.csv — ver Analisis_A1_A2_A3_Ground_Truth_Circularity.md
y el análisis de longitud de este TFM):

  Nivel 0 (oración):       SENTENCE_BOUNDARY_REGEX (.!?\n)
  Nivel 1 (cláusula ';'):  CLAUSE_BOUNDARY_SEMICOLON_REGEX
  Nivel 2 (cláusula ','):  CLAUSE_BOUNDARY_COMMA_REGEX

Una unidad solo escala al siguiente nivel cuando, tras agotar el nivel
actual, queda una pieza irreducible (una única unidad atómica de ese nivel)
que por sí sola sigue excediendo el umbral. Esto mantiene la intervención
acotada al residuo problemático, no al corpus completo.

Si una unidad agota los 3 niveles y sigue sin poder dividirse (sin ';' ni
','), se deja tal cual, marcada con metodo_division="sin_dividir_sin_delimitador".
No hay corte mecánico de último recurso (decisión explícita: estos casos son
raros -1 en 11.044 fragmentos verificados- y se tratan como excepción
documentada, no forzada).
"""

import re
from dataclasses import dataclass

from src.config.text_standards import (
    SENTENCE_BOUNDARY_REGEX,
    CLAUSE_BOUNDARY_SEMICOLON_REGEX,
    CLAUSE_BOUNDARY_COMMA_REGEX,
)

# Nombre de nivel -> regex. Orden = orden de escalado.
_NIVELES_CASCADA = [
    ("oracion", SENTENCE_BOUNDARY_REGEX),
    ("clausula_punto_y_coma", CLAUSE_BOUNDARY_SEMICOLON_REGEX),
    ("clausula_coma", CLAUSE_BOUNDARY_COMMA_REGEX),
]

_METODO_SIN_DIVIDIR = "sin_dividir"
_METODO_SIN_DELIMITADOR = "sin_dividir_sin_delimitador"


@dataclass
class _UnidadAtomica:
    """Una porción de texto extraída con el regex de un nivel, con su offset absoluto y su conteo de tokens."""
    texto: str
    start: int
    end: int
    tokens: int


def rebalancear_unidad(texto: str, offset_inicio: int, tokenizer, umbral: int) -> list[dict]:
    """
    Punto de entrada público. Divide `texto` en piezas <= `umbral` tokens
    cuando es posible, preservando offsets absolutos.

    Parámetros
    ----------
    texto : str
        Texto completo de la unidad a evaluar (un span de train o un
        fragmento de inferencia ya generado).
    offset_inicio : int
        Offset absoluto (dentro del documento original) donde empieza `texto`.
    tokenizer : AutoTokenizer
        Mismo tokenizer usado en el resto del pipeline (bert-base-uncased).
    umbral : int
        FRAGMENT_REBALANCE_THRESHOLD.

    Retorna
    -------
    list[dict]
        Uno o más elementos, cada uno:
            {
                "texto": str,
                "start_offset": int,
                "end_offset": int,
                "tokens": int,
                "metodo_division": str,
            }
        Si `texto` ya cabe en `umbral`, devuelve una lista de 1 elemento
        con metodo_division="sin_dividir".
    """
    tokens_totales = _contar_tokens(texto, tokenizer)

    if tokens_totales <= umbral:
        return [{
            "texto": texto,
            "start_offset": offset_inicio,
            "end_offset": offset_inicio + len(texto),
            "tokens": tokens_totales,
            "metodo_division": _METODO_SIN_DIVIDIR,
        }]

    piezas = _procesar_nivel(texto, offset_inicio, tokenizer, umbral, nivel_idx=0)
    return [
        {
            "texto": p.texto,
            "start_offset": p.start,
            "end_offset": p.end,
            "tokens": p.tokens,
            "metodo_division": metodo,
        }
        for p, metodo in piezas
    ]


def _contar_tokens(texto: str, tokenizer) -> int:
    return len(tokenizer.encode(texto, add_special_tokens=False))


def _extraer_unidades_atomicas(texto: str, offset_base: int, tokenizer, regex: str) -> list[_UnidadAtomica]:
    """
    Aplica `regex` sobre `texto` y devuelve las unidades resultantes con
    offset absoluto (offset_base + posición relativa) y tokens ya contados.

    IMPORTANTE: nunca descarta una coincidencia, aunque sea solo espacio en
    blanco (ej. el espacio entre el punto final de una oración y el inicio
    de la siguiente). Descartar esas coincidencias -como hace el filtro de
    calidad al validar fragmentos finales- rompería aquí la invariante de
    que la concatenación de unidades debe reconstruir el texto original
    byte a byte, y con ello desalinearía los offsets absolutos usados
    después para el cruce espacial contra las anotaciones de CUAD. Una
    coincidencia vacía tras strip() se fusiona con la unidad anterior en
    vez de crear una unidad nueva sin contenido útil.
    """
    unidades: list[_UnidadAtomica] = []
    for match in re.finditer(regex, texto):
        fragmento = match.group(0)
        start_abs = offset_base + match.start()
        end_abs = offset_base + match.end()

        if not fragmento.strip() and unidades:
            unidades[-1].texto += fragmento
            unidades[-1].end = end_abs
            continue

        unidades.append(_UnidadAtomica(
            texto=fragmento,
            start=start_abs,
            end=end_abs,
            tokens=_contar_tokens(fragmento, tokenizer),
        ))
    return unidades


def _agrupar_balanceado(unidades: list[_UnidadAtomica], umbral: int) -> list[list[_UnidadAtomica]]:
    """
    Agrupa unidades atómicas consecutivas en k=ceil(total_tokens/umbral)
    grupos, recalculando el objetivo por grupo tras cerrar cada uno:
        objetivo_actual = tokens_restantes / grupos_restantes

    A diferencia del algoritmo greedy actual (_split_long_paragraph_deterministically
    en fragmentation_utils.py, que llena hasta el tope y dejar el resto en
    el último chunk), este reparto se mantiene uniforme sin importar cuántas
    unidades atómicas haya.

    Una unidad atómica individual con tokens > umbral se aísla como grupo
    de tamaño 1 (no se combina con nada) — queda marcada para escalar al
    siguiente nivel de la cascada por el llamador.

    Garantiza devolver exactamente len(unidades) unidades distribuidas
    (ninguna se pierde ni se duplica).
    """
    total_tokens = sum(u.tokens for u in unidades)
    if total_tokens <= umbral:
        return [list(unidades)]

    k = -(-total_tokens // umbral)  # ceil sin importar math.ceil
    pendientes = list(unidades)
    grupos: list[list[_UnidadAtomica]] = []
    tokens_restantes = total_tokens
    grupos_restantes = k

    while pendientes:
        if grupos_restantes <= 1:
            grupos.append(pendientes)
            break

        objetivo = tokens_restantes / grupos_restantes
        grupo_actual: list[_UnidadAtomica] = []
        tokens_grupo = 0

        while pendientes:
            candidata = pendientes[0]
            if not grupo_actual:
                # Siempre se toma al menos una unidad, incluso si por sí
                # sola excede el objetivo (evita grupos vacíos / bucle infinito).
                grupo_actual.append(pendientes.pop(0))
                tokens_grupo += candidata.tokens
                continue
            if tokens_grupo + candidata.tokens <= objetivo:
                grupo_actual.append(pendientes.pop(0))
                tokens_grupo += candidata.tokens
            else:
                break

        grupos.append(grupo_actual)
        tokens_restantes -= tokens_grupo
        grupos_restantes -= 1

    return grupos


def _fusionar_grupo(grupo: list[_UnidadAtomica]) -> _UnidadAtomica:
    """Concatena las unidades de un grupo en una sola, con offsets extremos y tokens ya sumados (evita retokenizar)."""
    texto = "".join(u.texto for u in grupo)
    return _UnidadAtomica(
        texto=texto,
        start=grupo[0].start,
        end=grupo[-1].end,
        tokens=sum(u.tokens for u in grupo),
    )


def _procesar_nivel(
    texto: str, offset_inicio: int, tokenizer, umbral: int, nivel_idx: int
) -> list[tuple[_UnidadAtomica, str]]:
    """
    Procesa `texto` (que ya se sabe que excede `umbral`) en el nivel
    `nivel_idx` de la cascada. Devuelve lista de (pieza, metodo_division).
    """
    if nivel_idx >= len(_NIVELES_CASCADA):
        # Agotados los 3 niveles: no hay delimitador que permita dividir más.
        unidad = _UnidadAtomica(
            texto=texto, start=offset_inicio, end=offset_inicio + len(texto),
            tokens=_contar_tokens(texto, tokenizer),
        )
        return [(unidad, _METODO_SIN_DELIMITADOR)]

    nombre_nivel, regex_nivel = _NIVELES_CASCADA[nivel_idx]
    unidades = _extraer_unidades_atomicas(texto, offset_inicio, tokenizer, regex_nivel)

    if len(unidades) <= 1:
        # El delimitador de este nivel no partió nada (0 o 1 coincidencias)
        # -> escalar directamente al siguiente nivel sobre el mismo texto.
        return _procesar_nivel(texto, offset_inicio, tokenizer, umbral, nivel_idx + 1)

    grupos = _agrupar_balanceado(unidades, umbral)

    resultado: list[tuple[_UnidadAtomica, str]] = []
    for grupo in grupos:
        pieza = _fusionar_grupo(grupo)
        if pieza.tokens <= umbral:
            resultado.append((pieza, nombre_nivel))
        elif len(grupo) == 1:
            # Unidad irreducible en este nivel: escalar solo esta unidad.
            resultado.extend(_procesar_nivel(pieza.texto, pieza.start, tokenizer, umbral, nivel_idx + 1))
        else:
            # Grupo multi-unidad que igual excede el umbral (desbalance
            # residual del reparto): se re-intenta balancear ESTE grupo
            # entre sus propias unidades, sin escalar de nivel todavía.
            subgrupos = _agrupar_balanceado(grupo, umbral)
            if len(subgrupos) == 1 and len(subgrupos[0]) == len(grupo):
                # No se pudo mejorar (mismo agrupamiento) -> escalar para evitar bucle infinito.
                resultado.extend(_procesar_nivel(pieza.texto, pieza.start, tokenizer, umbral, nivel_idx + 1))
            else:
                for subgrupo in subgrupos:
                    subpieza = _fusionar_grupo(subgrupo)
                    if subpieza.tokens <= umbral:
                        resultado.append((subpieza, nombre_nivel))
                    else:
                        resultado.extend(_procesar_nivel(subpieza.texto, subpieza.start, tokenizer, umbral, nivel_idx + 1))

    return resultado