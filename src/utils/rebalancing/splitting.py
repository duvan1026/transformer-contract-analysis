"""
src/utils/rebalancing/splitting.py

Núcleo puro del rebalanceo de unidades de texto largas (spans de train o
fragmentos de inferencia). No depende de pandas ni de rutas de disco: opera
sobre (texto, offset_absoluto_de_inicio, tokenizer, objetivo, límite físico)
y devuelve una lista de piezas con sus offsets absolutos recalculados.

Distinción central de este módulo entre dos umbrales, que antes eran uno solo:

  - OBJETIVO (target_tokens, 250): decide EN CUÁNTOS grupos se reparte una
    unidad y qué tan parejos quedan esos grupos. Es un número de referencia
    para el reparto, no una prohibición: un grupo que queda un poco por
    encima del objetivo se acepta sin problema.

  - LÍMITE FÍSICO (max_tokens, 512): el único motivo real para forzar una
    división. Solo cuando una pieza irreducible (una unidad que ningún
    delimitador logra partir más) supera este número se escala al siguiente
    nivel de la cascada. Antes, un único umbral cumplía ambos roles, lo que
    forzaba a dividir bloques de texto perfectamente seguros para el modelo
    solo por estar un poco por encima del objetivo.

Estrategia (cascada de 2 niveles, la anterior tenía 3):

  Nivel 0 (oración):      SENTENCE_BOUNDARY_REGEX (.!?\n)
  Nivel 1 (cláusula ';'): CLAUSE_BOUNDARY_SEMICOLON_REGEX

Se escala de un nivel al siguiente únicamente cuando, dentro del nivel
actual, aparece una unidad irreducible (no hay más de una coincidencia del
delimitador) que por sí sola supera el límite físico. Si esa unidad está
por debajo del límite físico -aunque supere el objetivo-, se acepta tal
cual: no hay ninguna razón para forzar un corte adicional.

Se retiró el nivel de comas que existía antes. Con la distinción
objetivo/límite físico, ese nivel casi nunca llegaba a activarse, y cuando
lo hacía, el riesgo de partir una referencia cruzada o un término legal a la
mitad superaba el beneficio de acercarse más al objetivo.

Si una unidad llega al final de la cascada (agotó oración y ';') y sigue sin
poder dividirse porque no tiene ningún delimitador disponible, se deja tal
cual, marcada con metodo_division="sin_dividir_sin_delimitador". No hay
corte mecánico de último recurso: es una excepción documentada, no forzada.
"""

import re
from dataclasses import dataclass

from src.config.text_standards import (
    SENTENCE_BOUNDARY_REGEX,
    CLAUSE_BOUNDARY_SEMICOLON_REGEX,
)

# Nombre de nivel -> regex. Orden = orden de escalado.
_NIVELES_CASCADA = [
    ("oracion", SENTENCE_BOUNDARY_REGEX),
    ("clausula_punto_y_coma", CLAUSE_BOUNDARY_SEMICOLON_REGEX),
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


def rebalancear_unidad(texto: str, offset_inicio: int, tokenizer,
                        max_tokens: int, umbral_objetivo: int) -> list[dict]:
    """
    Punto de entrada público. Divide `texto` en piezas equilibradas cuando
    excede `target_tokens`, preservando offsets absolutos, y garantiza que
    ninguna pieza final supere `max_tokens` salvo el caso documentado de
    "sin_dividir_sin_delimitador".

    Parámetros
    ----------
    texto : str
        Texto completo de la unidad a evaluar (un span de train o un
        fragmento de inferencia ya generado).
    offset_inicio : int
        Offset absoluto (dentro del documento original) donde empieza `texto`.
    tokenizer : AutoTokenizer
        Mismo tokenizer usado en el resto del pipeline (bert-base-uncased).
    target_tokens : int
        FRAGMENT_REBALANCE_TARGET_TOKENS. Decide si se intenta repartir y en
        cuántos grupos.
    max_tokens : int
        FRAGMENT_REBALANCE_MAX_TOKENS. Único motivo real para forzar un
        corte adicional dentro de la cascada.

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
        Si `texto` ya cabe en `target_tokens`, devuelve una lista de 1
        elemento con metodo_division="sin_dividir".
    """
    tokens_totales = _contar_tokens(texto, tokenizer)

    if tokens_totales <= max_tokens:
        return [{
            "texto": texto,
            "start_offset": offset_inicio,
            "end_offset": offset_inicio + len(texto),
            "tokens": tokens_totales,
            "metodo_division": _METODO_SIN_DIVIDIR,
        }]

    piezas = _procesar_nivel(texto, offset_inicio, tokenizer, max_tokens, umbral_objetivo, nivel_idx=0)
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

    Nunca descarta una coincidencia, aunque sea solo espacio en blanco (ej.
    el espacio entre el punto final de una oración y el inicio de la
    siguiente): descartarla rompería la invariante de que la concatenación
    de unidades reconstruye el texto original byte a byte, desalineando los
    offsets absolutos usados después para el cruce espacial contra las
    anotaciones. Una coincidencia vacía tras strip() se fusiona con la
    unidad anterior en vez de crear una unidad nueva sin contenido útil.
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


def _agrupar_balanceado(unidades: list[_UnidadAtomica], target_tokens: int) -> list[list[_UnidadAtomica]]:
    """
    Agrupa unidades atómicas consecutivas en k=ceil(total_tokens/target_tokens)
    grupos, recalculando el objetivo por grupo tras cerrar cada uno:
        objetivo_actual = tokens_restantes / grupos_restantes

    Reparto uniforme sin importar cuántas unidades atómicas haya, en vez de
    llenar cada grupo hasta el tope y dejar el resto en el último.

    Una unidad atómica individual con tokens mayores al objetivo se agrupa
    igual con lo que haya alrededor -- que un grupo quede algo por encima
    del objetivo no es un problema: solo importa si termina superando el
    límite físico, y eso se decide después, en _procesar_nivel.

    Garantiza devolver exactamente len(unidades) unidades distribuidas
    (ninguna se pierde ni se duplica).
    """
    total_tokens = sum(u.tokens for u in unidades)
    if total_tokens <= target_tokens:
        return [list(unidades)]

    k = -(-total_tokens // target_tokens)  # ceil sin importar math.ceil
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
    texto: str, offset_inicio: int, tokenizer, max_tokens: int, umbral_objetivo: int, nivel_idx: int
) -> list[tuple[_UnidadAtomica, str]]:
    """
    Procesa `texto` (que ya se sabe que excede target_tokens) en el nivel
    `nivel_idx` de la cascada. Devuelve lista de (pieza, metodo_division).
    """
    if nivel_idx >= len(_NIVELES_CASCADA):
        # Agotados los niveles disponibles: no hay delimitador que permita
        # dividir más. Solo se llega aquí si la unidad ya supera max_tokens
        # (ver más abajo), así que queda marcada como excepción documentada.
        unidad = _UnidadAtomica(
            texto=texto, start=offset_inicio, end=offset_inicio + len(texto),
            tokens=_contar_tokens(texto, tokenizer),
        )
        return [(unidad, _METODO_SIN_DELIMITADOR)]

    nombre_nivel, regex_nivel = _NIVELES_CASCADA[nivel_idx]
    unidades = _extraer_unidades_atomicas(texto, offset_inicio, tokenizer, regex_nivel)

    if len(unidades) <= 1:
        return _procesar_nivel(texto, offset_inicio, tokenizer, max_tokens, umbral_objetivo, nivel_idx + 1)

    grupos = _agrupar_balanceado(unidades, umbral_objetivo)

    resultado: list[tuple[_UnidadAtomica, str]] = []
    for grupo in grupos:
        pieza = _fusionar_grupo(grupo)
        if pieza.tokens <= max_tokens:
            resultado.append((pieza, nombre_nivel))

        elif len(grupo) == 1:
            resultado.extend(_procesar_nivel(pieza.texto, pieza.start, tokenizer, max_tokens, umbral_objetivo, nivel_idx + 1))

        else:
            subgrupos = _agrupar_balanceado(grupo, umbral_objetivo)
            if len(subgrupos) == 1 and len(subgrupos[0]) == len(grupo):
                # No se pudo mejorar (mismo agrupamiento) -> escalar para evitar bucle infinito.
                resultado.extend(_procesar_nivel(pieza.texto, pieza.start, tokenizer, max_tokens, umbral_objetivo, nivel_idx + 1))

            else:
                for subgrupo in subgrupos:
                    subpieza = _fusionar_grupo(subgrupo)
                    if subpieza.tokens <= max_tokens:
                        resultado.append((subpieza, nombre_nivel))
                    else:
                        resultado.extend(_procesar_nivel(subpieza.texto, subpieza.start, tokenizer, max_tokens, umbral_objetivo, nivel_idx + 1))

    return resultado