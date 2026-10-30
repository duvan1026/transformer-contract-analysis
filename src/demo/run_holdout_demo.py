#!/usr/bin/env python3
""" 
Orquesta una demostración de inferencia sobre los mismos contratos para:
1. baseline_holdout
2. balanced_holdout

Uso desde la raíz del proyecto:
    python src/demo/run_holdout_demo.py "NOMBRE COMPLETO DEL CONTRATO"
    python src/demo/run_holdout_demo.py "CONTRATO 1; CONTRATO 2"

Los nombres se comparan sin distinguir mayúsculas/minúsculas y eliminando
espacios al principio y al final. No se exige ni se añade una extensión.
Para varios contratos, use punto y coma como separador, porque los nombres
pueden contener comas.

Antes de modificar cualquier registro, el script valida los contratos en
AMBOS CSV. Después ejecuta las cinco etapas de baseline_holdout y, solo si
terminan correctamente, las cinco etapas de balanced_holdout. Cada experimento
usa su propio registro y sus propias rutas de salida.
"""

from __future__ import annotations

import argparse
import contextlib
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

# Garantizar que los imports src.* funcionen al ejecutar este archivo directamente.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.paths import HOLDOUT_REGISTRY_BASELINE, HOLDOUT_REGISTRY_BALANCED
from src.config.text_standards import BATCH_SIZE_DOCS

AVAILABLE = "Disponible"
NONE_VALUE = "Ninguno"
TEMP_RESERVED = "__RESERVADO_TEMPORAL_DEMO__"
REGISTRY_COLUMNS = ("id_documento", "estado", "experimento", "split_asignado")
STATE_COLUMNS = ("estado", "experimento", "split_asignado")

EXPERIMENTS = (
    {
        "name": "baseline_holdout",
        "registry_path": HOLDOUT_REGISTRY_BASELINE,
        "steps": (
            ("Preparar lote de documentos", "src/inference/01_prepare_batch_documents.py"),
            ("Rebalancear fragmentos", "src/inference/01b_rebalance_fragments.py"),
            ("Ejecutar inferencia", "src/inference/02_run_inference.py"),
            ("Calcular Ground Truth", "src/inference/03_calculate_ground_truth.py"),
            ("Generar informes", "src/inference/04_generate_reports.py"),
        ),
    },
    {
        "name": "balanced_holdout",
        "registry_path": HOLDOUT_REGISTRY_BALANCED,
        "steps": (
            ("Preparar lote de documentos", "src/inference/01_prepare_batch_documents.py"),
            ("Rebalancear fragmentos", "src/inference/01b_rebalance_fragments.py"),
            ("Ejecutar inferencia", "src/inference/02_run_inference.py"),
            ("Calcular Ground Truth", "src/inference/03_calculate_ground_truth.py"),
            ("Generar informes", "src/inference/04_generate_reports.py"),
        ),
    },
)


class Tee:
    """Escribe simultáneamente en consola y en el archivo de log."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, message: str) -> int:
        for stream in self.streams:
            stream.write(message)
            stream.flush()
        return len(message)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def normalize_document_name(name: Any) -> str:
    """Normaliza solo para comparar; no modifica el valor almacenado en el CSV."""
    return str(name).strip().upper()


def read_and_validate_registry(registry_path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Carga un registro y crea un mapa ID normalizado -> ID original."""
    if not registry_path.exists():
        raise FileNotFoundError(
            f"[ERROR] No se encontró el registro holdout: {registry_path}"
        )

    df = pd.read_csv(registry_path, dtype={"id_documento": str})
    missing_columns = [column for column in REGISTRY_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"[ERROR] El registro {registry_path} no contiene las columnas requeridas: "
            + ", ".join(missing_columns)
        )

    normalized = df["id_documento"].map(
        lambda value: normalize_document_name(value) if pd.notna(value) else None
    )
    duplicated_mask = normalized.notna() & normalized.duplicated(keep=False)
    if duplicated_mask.any():
        duplicated = (
            df.loc[duplicated_mask, "id_documento"].astype(str).unique().tolist()
        )
        raise ValueError(
            f"[ERROR] El registro {registry_path} contiene identificadores duplicados "
            "después de normalizar mayúsculas y espacios externos: "
            + ", ".join(duplicated)
        )

    normalized_to_original = {
        normalized_id: original_id
        for normalized_id, original_id in zip(normalized, df["id_documento"])
        if normalized_id is not None
    }
    return df, normalized_to_original


def parse_file_names(raw: str, valid_normalized_ids: set[str]) -> list[str]:
    """Interpreta uno o varios nombres; devuelve nombres normalizados únicos."""
    raw_normalized = normalize_document_name(raw)
    if not raw_normalized:
        raise ValueError("Debes indicar al menos un nombre de contrato.")

    # Primero se comprueba el argumento completo: así se permiten comas dentro
    # del nombre de un único contrato sin dividirlo accidentalmente.
    if raw_normalized in valid_normalized_ids:
        return [raw_normalized]

    # Para varios contratos, el separador inequívoco es el punto y coma.
    if ";" in raw:
        names = [normalize_document_name(part) for part in raw.split(";")]
        names = [name for name in names if name]
    else:
        # Compatibilidad con listas separadas por comas: se intenta segmentar
        # únicamente cuando los identificadores registrados permiten resolverlas.
        remaining = raw_normalized
        names: list[str] = []
        sorted_ids = sorted(valid_normalized_ids, key=len, reverse=True)
        while remaining:
            candidates = [
                doc_id for doc_id in sorted_ids
                if remaining.startswith(doc_id)
                and (
                    len(remaining) == len(doc_id)
                    or remaining[len(doc_id)] == ","
                )
            ]
            if not candidates:
                # Se devuelve el argumento entero para producir un error claro
                # durante la validación conjunta de ambos registros.
                return [raw_normalized]
            chosen = candidates[0]
            names.append(chosen)
            remaining = remaining[len(chosen):]
            if remaining:
                remaining = remaining[1:].strip()
                if not remaining:
                    raise ValueError("La lista termina con un separador sin contrato.")

    if not names:
        raise ValueError("Debes indicar al menos un nombre de contrato.")
    if len(names) != len(set(names)):
        raise ValueError("La lista contiene contratos duplicados después de normalizar.")
    if any("/" in name or "\\" in name for name in names):
        raise ValueError("Indica el id_documento, no una ruta de archivo.")
    return names


def validate_requested_contracts(
    requested_normalized: list[str],
    registry_maps: dict[str, dict[str, str]],
) -> dict[str, list[str]]:
    """Exige que todos los contratos existan en ambos registros antes de cambiar nada."""
    missing_by_experiment: dict[str, list[str]] = {}
    resolved_by_experiment: dict[str, list[str]] = {}

    for experiment in EXPERIMENTS:
        name = experiment["name"]
        id_map = registry_maps[name]
        missing = [contract for contract in requested_normalized if contract not in id_map]
        if missing:
            missing_by_experiment[name] = missing
        else:
            # Usar los identificadores originales de cada CSV en las operaciones posteriores.
            resolved_by_experiment[name] = [id_map[item] for item in requested_normalized]

    if missing_by_experiment:
        details = []
        for experiment_name, missing in missing_by_experiment.items():
            registry_path = next(
                item["registry_path"] for item in EXPERIMENTS
                if item["name"] == experiment_name
            )
            for contract in missing:
                details.append(
                    f"  - '{contract}' no se encuentra en {registry_path} "
                    "(columna id_documento)."
                )
        raise ValueError(
            "[ERROR] No se modificó ningún registro y no se ejecutó ningún experimento. "
            "Todos los contratos deben existir en ambos registros.\n"
            + "\n".join(details)
        )

    if len(requested_normalized) > BATCH_SIZE_DOCS:
        raise ValueError(
            f"[ERROR] Has indicado {len(requested_normalized)} contratos, pero "
            f"BATCH_SIZE_DOCS={BATCH_SIZE_DOCS}. Reduce la lista o ajusta explícitamente "
            "el tamaño de lote. No se modificó ningún registro."
        )

    return resolved_by_experiment


def prepare_registry_for_demo(
    df_original: pd.DataFrame,
    target_ids: list[str],
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Libera objetivos y reserva temporalmente candidatos ajenos al lote."""
    df = df_original.copy()
    target_set = set(target_ids)

    # El preparador selecciona filas con los tres campos en estado inicial.
    eligible_mask = (
        (df["estado"] == AVAILABLE)
        & (df["experimento"] == NONE_VALUE)
        & (df["split_asignado"] == NONE_VALUE)
    )
    non_target_mask = ~df["id_documento"].astype(str).isin(target_set)
    reserve_mask = eligible_mask & non_target_mask

    # Guardar exclusivamente los campos que el script cambia temporalmente.
    snapshot: dict[str, dict[str, Any]] = {}
    for _, row in df.loc[reserve_mask].iterrows():
        document_id = str(row["id_documento"])
        snapshot[document_id] = {column: row[column] for column in STATE_COLUMNS}

    # Evitar que la selección aleatoria incluya contratos no solicitados.
    df.loc[reserve_mask, "estado"] = TEMP_RESERVED

    # Restablecer los contratos solicitados al estado inicial de procesamiento.
    target_mask = df["id_documento"].astype(str).isin(target_set)
    df.loc[target_mask, "estado"] = AVAILABLE
    df.loc[target_mask, "experimento"] = NONE_VALUE
    df.loc[target_mask, "split_asignado"] = NONE_VALUE

    return df, snapshot


def restore_non_target_fields(
    registry_path: Path,
    snapshot: dict[str, dict[str, Any]],
) -> None:
    """Restaura los campos temporalmente reservados sin revertir el estado objetivo."""
    if not snapshot:
        return
    if not registry_path.exists():
        raise FileNotFoundError(f"No se puede restaurar el registro ausente: {registry_path}")

    current = pd.read_csv(registry_path, dtype={"id_documento": str})
    if "id_documento" not in current.columns:
        raise ValueError(f"No se puede restaurar {registry_path}: falta id_documento.")

    for document_id, values in snapshot.items():
        mask = current["id_documento"].astype(str) == document_id
        if mask.any():
            for column, value in values.items():
                current.loc[mask, column] = value

    current.to_csv(registry_path, index=False, encoding="utf-8")
    print(f"[+] Restaurados los estados temporales de documentos ajenos al lote: {registry_path}")


def run_step(experiment_name: str, step_name: str, script_path: str) -> None:
    """Ejecuta una etapa y propaga cualquier error para detener la secuencia."""
    command = [sys.executable, script_path, "--model", experiment_name]
    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    start = time.monotonic()

    print("-" * 78)
    print(f"[{started_at}] INICIO {experiment_name} | {step_name}")
    print("Comando:", " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)

    duration = int(time.monotonic() - start)
    finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{finished_at}] OK {experiment_name} | {step_name}")
    print(f"Duración: {duration // 60}m {duration % 60}s ({duration}s)")
    print("-" * 78)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta baseline_holdout y balanced_holdout, en ese orden, "
            "sobre los mismos contratos identificados por id_documento."
        )
    )
    parser.add_argument(
        "files",
        help=(
            "Nombre de un contrato o varios nombres separados por punto y coma. "
            "Los nombres pueden contener comas y no necesitan extensión."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    # Fase 1: cargar y validar ambos registros antes de modificar cualquiera.
    try:
        registry_frames: dict[str, pd.DataFrame] = {}
        registry_maps: dict[str, dict[str, str]] = {}
        all_normalized_ids: set[str] = set()

        for experiment in EXPERIMENTS:
            name = experiment["name"]
            df, id_map = read_and_validate_registry(experiment["registry_path"])
            registry_frames[name] = df
            registry_maps[name] = id_map
            all_normalized_ids.update(id_map)

        requested_normalized = parse_file_names(args.files, all_normalized_ids)
        resolved_ids = validate_requested_contracts(requested_normalized, registry_maps)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    # Crear un único log para toda la demostración de los dos experimentos.
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"holdout_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    total_start = time.monotonic()
    exit_code = 0
    snapshots: dict[str, dict[str, dict[str, Any]]] = {}

    print("=" * 78)
    print("DEMOSTRACIÓN DE INFERENCIA NLP: BASELINE HOLDOUT → BALANCED HOLDOUT")
    print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Contratos solicitados: {len(requested_normalized)}")
    for name in requested_normalized:
        print(f"  - {name}")
    print(f"Log: {log_path}")
    print("=" * 78)

    with open(log_path, "a", encoding="utf-8") as log_file:
        tee_out = Tee(sys.stdout, log_file)
        tee_err = Tee(sys.stderr, log_file)
        try:
            with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
                for experiment in EXPERIMENTS:
                    experiment_name = experiment["name"]
                    registry_path: Path = experiment["registry_path"]
                    target_ids = resolved_ids[experiment_name]

                    print("\n" + "#" * 78)
                    print(f"INICIANDO EXPERIMENTO: {experiment_name}")
                    print(f"Registro: {registry_path}")
                    print("#" * 78)

                    # Releer el CSV justo antes de prepararlo para no trabajar con
                    # una copia desactualizada si el pipeline anterior lo modificó.
                    current_df, _ = read_and_validate_registry(registry_path)
                    prepared_df, snapshot = prepare_registry_for_demo(current_df, target_ids)
                    snapshots[experiment_name] = snapshot
                    prepared_df.to_csv(registry_path, index=False, encoding="utf-8")
                    print(
                        "[+] Registro preparado para los contratos solicitados: "
                        "estado='Disponible', experimento='Ninguno', split_asignado='Ninguno'."
                    )

                    # Cada etapa recibe el modo del experimento correspondiente.
                    # Un fallo detiene la secuencia; balanced no empieza si baseline falla.
                    for step_name, script_path in experiment["steps"]:
                        run_step(experiment_name, step_name, script_path)

                    print(f"[+] Experimento {experiment_name} completado.")

        except subprocess.CalledProcessError as exc:
            exit_code = exc.returncode or 1
            print(
                f"[ERROR] Una etapa terminó con código {exit_code}. "
                "Se detiene la secuencia y no se iniciarán los experimentos restantes.",
                file=sys.stderr,
            )
        except Exception:
            exit_code = 1
            traceback.print_exc(file=sys.stderr)
        finally:
            # Restaurar los documentos ajenos al lote en ambos registros, incluso
            # si alguna etapa falla. No se revierten los estados de los objetivos.
            for experiment in EXPERIMENTS:
                experiment_name = experiment["name"]
                registry_path = experiment["registry_path"]
                try:
                    restore_non_target_fields(registry_path, snapshots.get(experiment_name, {}))
                except Exception:
                    exit_code = exit_code or 1
                    print(
                        f"[ERROR] No se pudo restaurar el registro {registry_path}. "
                        "Revísalo manualmente.",
                        file=sys.stderr,
                    )
                    traceback.print_exc(file=sys.stderr)

    total_duration = int(time.monotonic() - total_start)
    print("=" * 78)
    if exit_code == 0:
        print("[OK] baseline_holdout y balanced_holdout finalizaron correctamente.")
    else:
        print(f"[ERROR] La demostración terminó con errores (código {exit_code}).")
    print(f"Duración total: {total_duration // 60}m {total_duration % 60}s ({total_duration}s)")
    print(f"Log: {log_path}")
    print("=" * 78)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
