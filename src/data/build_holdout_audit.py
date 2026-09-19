"""
src/data/build_holdout_audit.py

FASE 1 del flujo de construcción del lote final de holdout (100 contratos
no vistos, comunes a Baseline y Balanced, para la evaluación definitiva
del TFM).

Este script es de SOLO LECTURA sobre los dos registros maestros existentes
(MASTER_REGISTRY_BASELINE, MASTER_REGISTRY_BALANCED). No los modifica, no
los reescribe, y NO invoca 00_init_master_registry.py bajo ninguna
circunstancia -- ese script reconstruye el maestro completo desde cero
basándose en el train/val/test vigente en el momento de su ejecución, y
usarlo aquí destruiría la trazabilidad histórica que este script necesita
leer intacta.

Construye data/metadata/final_holdout_audit.csv: una tabla documento x
{baseline, balanced} que cruza el estado de cada id_documento en ambos
carriles, y determina si debe excluirse del universo candidato al lote
final de holdout.

Criterio de exclusión (excluido_para_holdout = True) para un id_documento:
    - estado_baseline == "Utilizado"   OR
    - estado_balanced == "Utilizado"   OR
    - estado_baseline == "En_proceso"  OR
    - estado_balanced == "En_proceso"

"Utilizado" (DOC_STATES["PROCESSED"]) es la marca genérica que el registro
maestro usa tanto para train/validation/test (vía split_asignado distinto
de "Ninguno") como para inferencias previas (vía experimento con prefijo
"v2_inference_..."). No es necesario desagregar por sub-etapa para decidir
la exclusión: cualquier valor "Utilizado" implica que el documento ya fue
procesado por el sistema en algún punto del desarrollo.

"En_proceso" (DOC_STATES["IN_PROGRESS"]) no debería aparecer en un
registro maestro "sano" en el momento de construir el holdout -- indicaría
una ejecución de 01_prepare_batch_documents.py interrumpida sin cierre
limpio (documento reservado pero nunca confirmado ni revertido). Si
aparece, este script emite una advertencia explícita por consola y, de
todas formas, excluye el documento: el sistema ya lo tocó y pudo haber
sido observado durante el desarrollo, que es justamente el criterio de
exclusión confirmado por el tutor.

Universo de referencia: el MISMO universo completo de CUAD que ya cubren
ambos registros maestros (00_init_master_registry.py no filtra por
categorías positivas, toma todos los "title" de CUAD_v1.json). Este script
NO reintroduce ningún filtro adicional de "documento con al menos un span
positivo bajo la taxonomía vigente" -- decisión registrada explícitamente
en el hilo de diseño: los lotes de inferencia ya generados por el pipeline
(01_prepare_batch_documents.py) tampoco aplican ese filtro, así que
aplicarlo aquí introduciría una discrepancia metodológica respecto a cómo
se han construido todos los lotes anteriores, y un sesgo optimista
respecto al comportamiento esperado en producción (donde el modelo debe
manejar también documentos sin ninguna cláusula relevante).

Uso:
    python src/data/build_holdout_audit.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config.paths import (
    MASTER_REGISTRY_BASELINE,
    MASTER_REGISTRY_BALANCED,
    HOLDOUT_AUDIT_FILE,
)
from src.config.text_standards import DOC_STATES

COL_ID = "id_documento"
COL_ESTADO = "estado"
COL_EXPERIMENTO = "experimento"
COL_SPLIT = "split_asignado"


def _cargar_maestro(path: Path, nombre_lane: str) -> pd.DataFrame:
    """
    Carga un registro maestro y renombra sus columnas con el sufijo del
    lane correspondiente, para poder cruzarlo después con el otro maestro
    sin colisión de nombres. Valida existencia, esquema y ausencia de
    id_documento duplicados antes de continuar -- un maestro corrupto no
    debe pasar en silencio a la fase de selección.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"[ERROR] No se encontró el registro maestro de '{nombre_lane}' en: {path}. "
            f"Este script requiere que ambos maestros (baseline y balanced) existan "
            f"y reflejen el estado real del histórico de ejecuciones."
        )
    df = pd.read_csv(path)

    columnas_requeridas = {COL_ID, COL_ESTADO, COL_EXPERIMENTO, COL_SPLIT}
    faltantes = columnas_requeridas - set(df.columns)
    if faltantes:
        raise KeyError(
            f"[ERROR] El registro maestro de '{nombre_lane}' ({path}) no tiene las "
            f"columnas esperadas. Faltan: {faltantes}"
        )

    duplicados = int(df[COL_ID].duplicated().sum())
    if duplicados > 0:
        raise ValueError(
            f"[ERROR] El registro maestro de '{nombre_lane}' ({path}) tiene "
            f"{duplicados} id_documento duplicados. No se puede construir la "
            f"auditoría sobre un maestro con duplicados; corrígelo manualmente "
            f"antes de continuar."
        )

    return df.rename(columns={
        COL_ESTADO: f"estado_{nombre_lane}",
        COL_EXPERIMENTO: f"experimento_{nombre_lane}",
        COL_SPLIT: f"split_{nombre_lane}",
    })


def _reportar_en_proceso(df_audit: pd.DataFrame) -> None:
    """
    Emite una advertencia explícita por consola si se detecta algún
    id_documento en estado 'En_proceso' en cualquiera de los dos maestros.
    No detiene la ejecución (el documento se excluye igualmente), pero
    deja constancia clara para revisión manual, tal como se acordó.
    """
    mask_en_proceso = (
        (df_audit["estado_baseline"] == DOC_STATES["IN_PROGRESS"]) |
        (df_audit["estado_balanced"] == DOC_STATES["IN_PROGRESS"])
    )
    n_en_proceso = int(mask_en_proceso.sum())

    if n_en_proceso > 0:
        print(
            f"\n[ADVERTENCIA] Se encontraron {n_en_proceso} id_documento en estado "
            f"'{DOC_STATES['IN_PROGRESS']}' en al menos uno de los dos registros "
            f"maestros. Esto indica una ejecución de 01_prepare_batch_documents.py "
            f"que quedó sin cerrar correctamente (el documento fue reservado pero "
            f"nunca se marcó como 'Utilizado' ni se revirtió a 'Disponible').\n"
            f"Estos documentos SE EXCLUYEN del universo candidato al holdout, "
            f"por el mismo criterio que un documento 'Utilizado': el sistema ya "
            f"lo tocó y pudo haber sido observado durante el desarrollo.\n"
            f"Revisa manualmente estos id_documento antes de continuar con la "
            f"Fase 2 (selección de los 100):\n"
        )
        print(
            df_audit.loc[
                mask_en_proceso,
                ["id_documento", "estado_baseline", "estado_balanced"]
            ].to_string(index=False)
        )
        print("")


def construir_audit() -> pd.DataFrame:
    """
    Construye la tabla de auditoría completa cruzando ambos maestros.
    """
    print("[*] Cargando registro maestro Baseline...")
    df_baseline = _cargar_maestro(MASTER_REGISTRY_BASELINE, "baseline")

    print("[*] Cargando registro maestro Balanced...")
    df_balanced = _cargar_maestro(MASTER_REGISTRY_BALANCED, "balanced")

    # Verificación de universo: ambos maestros deben cubrir exactamente el
    # mismo conjunto de id_documento (ambos se construyen desde el mismo
    # CUAD_JSON completo, sin filtrar, según 00_init_master_registry.py).
    # Si no coincide, es señal de que uno de los dos no está actualizado o
    # corresponde a una versión distinta de CUAD -- no se debe continuar
    # en silencio ante esa discrepancia.
    ids_baseline = set(df_baseline[COL_ID])
    ids_balanced = set(df_balanced[COL_ID])

    if ids_baseline != ids_balanced:
        solo_baseline = ids_baseline - ids_balanced
        solo_balanced = ids_balanced - ids_baseline
        raise ValueError(
            f"[ERROR] Los universos de id_documento de ambos maestros NO coinciden. "
            f"Documentos solo en baseline: {len(solo_baseline)}. "
            f"Documentos solo en balanced: {len(solo_balanced)}. "
            f"Esto indica que uno de los dos maestros no refleja el mismo "
            f"CUAD_v1.json que el otro, o que uno está desactualizado. "
            f"Debe resolverse manualmente antes de continuar -- no se genera "
            f"la auditoría con universos inconsistentes."
        )

    print(f"[*] Universo verificado: {len(ids_baseline)} id_documento, "
          f"idéntico entre ambos maestros.")

    df_audit = pd.merge(df_baseline, df_balanced, on=COL_ID, how="outer")

    df_audit["usado_en_baseline"] = df_audit["estado_baseline"] == DOC_STATES["PROCESSED"]
    df_audit["usado_en_balanced"] = df_audit["estado_balanced"] == DOC_STATES["PROCESSED"]

    en_proceso_baseline = df_audit["estado_baseline"] == DOC_STATES["IN_PROGRESS"]
    en_proceso_balanced = df_audit["estado_balanced"] == DOC_STATES["IN_PROGRESS"]

    df_audit["excluido_para_holdout"] = (
        df_audit["usado_en_baseline"] |
        df_audit["usado_en_balanced"] |
        en_proceso_baseline |
        en_proceso_balanced
    )

    def _motivo(row) -> str:
        motivos = []
        if row["usado_en_baseline"]:
            motivos.append("Utilizado_baseline")
        if row["usado_en_balanced"]:
            motivos.append("Utilizado_balanced")
        if row["estado_baseline"] == DOC_STATES["IN_PROGRESS"]:
            motivos.append("En_proceso_baseline")
        if row["estado_balanced"] == DOC_STATES["IN_PROGRESS"]:
            motivos.append("En_proceso_balanced")
        return ";".join(motivos) if motivos else ""

    df_audit["motivo_exclusion"] = df_audit.apply(_motivo, axis=1)

    columnas_finales = [
        "id_documento",
        "estado_baseline", "experimento_baseline", "split_baseline", "usado_en_baseline",
        "estado_balanced", "experimento_balanced", "split_balanced", "usado_en_balanced",
        "excluido_para_holdout", "motivo_exclusion",
    ]
    df_audit = df_audit[columnas_finales].sort_values("id_documento").reset_index(drop=True)

    _reportar_en_proceso(df_audit)

    return df_audit


def main():
    print("[*] Iniciando construcción de la auditoría de holdout (Fase 1)...")
    print("    Este script es de SOLO LECTURA sobre los registros maestros. "
          "No los modifica ni invoca 00_init_master_registry.py.\n")

    df_audit = construir_audit()

    n_total = len(df_audit)
    n_excluidos = int(df_audit["excluido_para_holdout"].sum())
    n_elegibles = n_total - n_excluidos

    print("\n" + "=" * 60)
    print("RESUMEN DE AUDITORÍA DE HOLDOUT")
    print("=" * 60)
    print(f"Total documentos en el universo (CUAD completo): {n_total}")
    print(f"Excluidos (usados previamente en baseline y/o balanced, "
          f"o en estado 'En_proceso'): {n_excluidos}")
    print(f"Elegibles como candidatos al lote final de holdout: {n_elegibles}")

    if n_elegibles < 100:
        print(
            f"\n[ADVERTENCIA CRÍTICA] Solo hay {n_elegibles} documentos elegibles, "
            f"menos de los 100 requeridos por el diseño experimental. "
            f"La Fase 2 (selección de los 100) fallará con este resultado. "
            f"Revisa el criterio de exclusión y el estado de los maestros "
            f"antes de continuar."
        )

    HOLDOUT_AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_audit.to_csv(HOLDOUT_AUDIT_FILE, index=False, encoding="utf-8")
    print(f"\n[+] Auditoría guardada en: {HOLDOUT_AUDIT_FILE}")


if __name__ == "__main__":
    main()