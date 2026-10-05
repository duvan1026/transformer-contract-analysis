"""
src/data/build_holdout_registry.py

FASE 2 del flujo de construcción del lote final de holdout (100 contratos
no vistos, comunes a Baseline y Balanced, para la evaluación definitiva
del TFM).

Este script NO lee los registros maestros directamente. Lee exclusivamente
la salida ya validada de la Fase 1 (HOLDOUT_AUDIT_FILE, generado por
build_holdout_audit.py), para que exista una única fuente de verdad sobre
qué documentos están excluidos -- evita que Fase 1 y Fase 2 puedan llegar
a conclusiones distintas si se ejecutan sobre estados de los maestros
ligeramente distintos en el tiempo.

Universo de candidatos: TODOS los id_documento con excluido_para_holdout
== False en el audit -- sin ningún filtro adicional de "documento con al
menos un span positivo bajo la taxonomía vigente".

Selección: aleatoria pero reproducible, con RANDOM_SEED. Se ejecuta UNA SOLA
VEZ para ambas lanes -- no hay dos muestreos independientes de Baseline y
Balanced: el resultado siempre se copia idéntico a los dos registros de
salida.

El orden de los candidatos se fija explícitamente con sorted() antes de
muestrear. Un set/Index de pandas no garantiza orden estable entre
ejecuciones o entornos, así que ordenar antes de aplicar
random_state=RANDOM_SEED es imprescindible para que la reproducibilidad
sea real y no dependa del hashing interno de la sesión de Python.

Salidas:
    - HOLDOUT_REGISTRY_BASELINE  (100 filas, esquema idéntico a los
      registros maestros: id_documento, estado, experimento, split_asignado)
    - HOLDOUT_REGISTRY_BALANCED  (mismas 100 filas, mismo esquema)

Ambos registros se generan con estado="Disponible", experimento="Ninguno",
split_asignado="Ninguno" -- el mismo estado "neutro" que tendría cualquier
documento recién incorporado al universo, porque en este punto los 100
documentos del holdout todavía no han pasado por ninguna etapa del
pipeline de inferencia (eso ocurre en la Fase 4, cuando
01_prepare_batch_documents.py --model {baseline_holdout|balanced_holdout}
los procese y actualice su estado a "Utilizado").

GUARDA DE NO-SOBRESCRITURA (crítica): este script se ejecuta UNA SOLA VEZ
en la vida del proyecto (vía "run_pipeline.sh build_holdout", fuera de
cualquier ejecución de baseline_holdout/balanced_holdout). Si se
reejecutara después de que baseline_holdout o balanced_holdout ya hayan
consumido y actualizado su propio registro operativo (marcándolo como
"Utilizado" en vez de "Disponible"), volvería a sobrescribirlo con el
estado "Disponible" original, borrando el rastro de que esa lane ya
procesó el lote y permitiendo un reprocesamiento redundante o
inconsistente. Por eso, si CUALQUIERA de los dos registros operativos ya
existe en disco, el script se detiene sin sobrescribir nada -- quien
quiera regenerar la selección a propósito debe borrar manualmente ambos
archivos primero, dejando constancia consciente de esa decisión.

Uso:
    python src/data/build_holdout_registry.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config.paths import (
    HOLDOUT_AUDIT_FILE,
    HOLDOUT_REGISTRY_BASELINE,
    HOLDOUT_REGISTRY_BALANCED,
)
from src.config.text_standards import DOC_STATES, RANDOM_SEED, N_HOLDOUT

COL_ID = "id_documento"

def _verificar_no_sobrescritura() -> None:
    """
    Guarda de seguridad: si cualquiera de los dos registros operativos ya
    existe, detiene la ejecución con un mensaje explícito en vez de
    sobrescribirlo en silencio.
    """
    existentes = [
        p for p in (HOLDOUT_REGISTRY_BASELINE, HOLDOUT_REGISTRY_BALANCED)
        if p.exists()
    ]
    if existentes:
        lista = "\n".join(f"  - {p}" for p in existentes)
        raise FileExistsError(
            f"[ERROR] Ya existe al menos uno de los registros operativos de "
            f"holdout:\n{lista}\n\n"
            f"Este script (build_holdout_registry.py) se ejecuta UNA SOLA VEZ "
            f"en la vida del proyecto. Volver a ejecutarlo ahora sobrescribiría "
            f"el estado de la selección -- si alguna de las dos lanes "
            f"(baseline_holdout / balanced_holdout) ya consumió este registro "
            f"y lo actualizó a 'Utilizado', esa marca se perdería.\n\n"
            f"Si de verdad quieres regenerar la selección desde cero (por "
            f"ejemplo, porque cambió el criterio de exclusión en la Fase 1), "
            f"borra manualmente ambos archivos primero y vuelve a ejecutar "
            f"este script de forma consciente:\n"
            f"  - {HOLDOUT_REGISTRY_BASELINE}\n"
            f"  - {HOLDOUT_REGISTRY_BALANCED}"
        )


def _cargar_audit() -> pd.DataFrame:
    if not HOLDOUT_AUDIT_FILE.exists():
        raise FileNotFoundError(
            f"[ERROR] No se encontró {HOLDOUT_AUDIT_FILE}. "
            f"Ejecuta primero: python src/data/build_holdout_audit.py"
        )

    df_audit = pd.read_csv(HOLDOUT_AUDIT_FILE)

    columnas_requeridas = {COL_ID, "excluido_para_holdout"}
    faltantes = columnas_requeridas - set(df_audit.columns)
    if faltantes:
        raise KeyError(
            f"[ERROR] {HOLDOUT_AUDIT_FILE} no tiene las columnas esperadas. "
            f"Faltan: {faltantes}. ¿Es realmente la salida de "
            f"build_holdout_audit.py (Fase 1)?"
        )

    duplicados = int(df_audit[COL_ID].duplicated().sum())
    if duplicados > 0:
        raise ValueError(
            f"[ERROR] {HOLDOUT_AUDIT_FILE} tiene {duplicados} id_documento "
            f"duplicados. La Fase 1 ya debería garantizar unicidad -- revisa "
            f"si el archivo fue editado manualmente o corresponde a una "
            f"ejecución corrupta de build_holdout_audit.py."
        )

    return df_audit


def seleccionar_holdout(df_audit: pd.DataFrame) -> list:
    """
    Selecciona N_HOLDOUT id_documento de forma aleatoria pero reproducible,
    a partir del pool de documentos elegibles (excluido_para_holdout ==
    False) del audit ya validado en la Fase 1.
    """
    documentos_elegibles = sorted(
        df_audit.loc[~df_audit["excluido_para_holdout"], COL_ID].tolist()
    )
    n_elegibles = len(documentos_elegibles)

    print(f"[*] Documentos elegibles según {HOLDOUT_AUDIT_FILE.name}: {n_elegibles}")

    if n_elegibles < N_HOLDOUT:
        raise ValueError(
            f"[ERROR] Solo hay {n_elegibles} documentos elegibles, menos de "
            f"los {N_HOLDOUT} requeridos por el diseño experimental. "
            f"No se puede completar la selección. Revisa el criterio de "
            f"exclusión en la Fase 1 antes de continuar."
        )

    seleccionados = (
        pd.Series(documentos_elegibles)
        .sample(n=N_HOLDOUT, random_state=RANDOM_SEED)
        .sort_values()
        .tolist()
    )

    assert len(seleccionados) == N_HOLDOUT, (
        f"[ERROR INTERNO] Se esperaban {N_HOLDOUT} documentos seleccionados, "
        f"se obtuvieron {len(seleccionados)}."
    )
    assert len(set(seleccionados)) == N_HOLDOUT, (
        "[ERROR INTERNO] La selección contiene id_documento duplicados."
    )

    print(f"[*] Selección reproducible completada: {N_HOLDOUT} documentos "
          f"(semilla RANDOM_SEED={RANDOM_SEED}).")

    return seleccionados


def construir_registro_operativo(seleccionados: list) -> pd.DataFrame:
    """
    Construye el DataFrame con el mismo esquema de 4 columnas que usan los
    registros maestros existentes (MASTER_REGISTRY_BASELINE/_BALANCED),
    para que 01_prepare_batch_documents.py pueda consumirlo sin cambios en
    su lógica interna de filtrado (estado=="Disponible" &
    experimento=="Ninguno" & split_asignado=="Ninguno").
    """
    return pd.DataFrame({
        "id_documento": seleccionados,
        "estado": DOC_STATES["AVAILABLE"],
        "experimento": DOC_STATES["NONE"],
        "split_asignado": DOC_STATES["NONE"],
    })


def main():
    print("[*] Iniciando construcción del registro operativo de holdout (Fase 2)...")
    print("    Fuente: salida ya validada de build_holdout_audit.py (Fase 1). "
          "No se leen los registros maestros directamente.\n")

    _verificar_no_sobrescritura()

    df_audit = _cargar_audit()
    n_elegibles_origen = int((~df_audit["excluido_para_holdout"]).sum())

    seleccionados = seleccionar_holdout(df_audit)

    df_registro = construir_registro_operativo(seleccionados)

    HOLDOUT_REGISTRY_BASELINE.parent.mkdir(parents=True, exist_ok=True)
    df_registro.to_csv(HOLDOUT_REGISTRY_BASELINE, index=False, encoding="utf-8")
    df_registro.to_csv(HOLDOUT_REGISTRY_BALANCED, index=False, encoding="utf-8")

    # Verificación explícita de paridad exacta entre ambos archivos antes
    # de darlos por generados -- deben ser el mismo contenido, no solo
    # "haber sido creados a partir de la misma lista" en teoría.
    ids_baseline = set(pd.read_csv(HOLDOUT_REGISTRY_BASELINE)["id_documento"])
    ids_balanced = set(pd.read_csv(HOLDOUT_REGISTRY_BALANCED)["id_documento"])
    assert ids_baseline == ids_balanced, (
        "[ERROR INTERNO] Los registros operativos de baseline y balanced "
        "no contienen exactamente los mismos id_documento tras guardarlos."
    )

    print(f"[+] Registro operativo guardado en: {HOLDOUT_REGISTRY_BASELINE}")
    print(f"[+] Registro operativo guardado en: {HOLDOUT_REGISTRY_BALANCED}")
    print(f"[+] Verificado: ambos registros contienen exactamente los mismos "
          f"{len(ids_baseline)} id_documento.")

    print("\n" + "=" * 60)
    print("RESUMEN DE SELECCIÓN DE HOLDOUT")
    print("=" * 60)
    print(f"Documentos elegibles de origen: {n_elegibles_origen}")
    print(f"Documentos seleccionados: {len(seleccionados)}")
    print(f"Semilla utilizada: {RANDOM_SEED}")


if __name__ == "__main__":
    main()