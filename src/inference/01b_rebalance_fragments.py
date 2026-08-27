"""
src/inference/01b_rebalance_fragments.py

Aplica el rebalanceo de fragmentos largos (src/utils/rebalancing/) a
batch_fragments.csv, sobrescribiendo el archivo en su misma ruta.

Se ejecuta DESPUÉS de 01_prepare_batch_documents.py y ANTES de
02_run_inference.py -- es imprescindible que vaya antes de la inferencia
neuronal: cada sub-fragmento resultante debe recibir su propio softmax real,
calculado sobre su propio texto más corto. Si este paso corriera después de
02_run_inference.py, se estarían reetiquetando trozos de texto que el
modelo nunca evaluó individualmente, invalidando el propósito del rebalanceo.

parrafo_id y fragmento_id originales se preservan sin renumerar; se añade
subfragmento_id como nuevo nivel de la jerarquía.
"""

import sys
import argparse
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config.paths import (
    FRAGMENTS_BASELINE,
    FRAGMENTS_BALANCED,
    REBALANCE_REJECTED_INFERENCE_BASELINE,
    REBALANCE_REJECTED_INFERENCE_BALANCED,
    REBALANCE_REPORT_INFERENCE_BASELINE,
    REBALANCE_REPORT_INFERENCE_BALANCED,
)
from src.config.text_standards import FRAGMENT_REBALANCE_THRESHOLD, MODEL_NAME
from src.utils.rebalancing.io import rebalancear_fragmentos_inferencia
from src.utils.rebalancing.stats import resumir_inferencia


def main(model_type):
    print(f"[*] Iniciando rebalanceo de fragmentos largos. MODO: {model_type.upper()}")

    if model_type == "balanced":
        fragments_path = FRAGMENTS_BALANCED
        rejected_path = REBALANCE_REJECTED_INFERENCE_BALANCED
        report_path = REBALANCE_REPORT_INFERENCE_BALANCED
    else:
        fragments_path = FRAGMENTS_BASELINE
        rejected_path = REBALANCE_REJECTED_INFERENCE_BASELINE
        report_path = REBALANCE_REPORT_INFERENCE_BASELINE

    if not fragments_path.exists():
        print(f"[ERROR] No se encontró {fragments_path}.")
        print(f"        Ejecuta primero: python src/inference/01_prepare_batch_documents.py --model {model_type}")
        return

    print(f"[*] Cargando tokenizador: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    df_fragmentos = pd.read_csv(fragments_path)
    n_antes = len(df_fragmentos)
    print(f"[*] Fragmentos cargados: {n_antes}")

    df_resultado, df_rechazadas = rebalancear_fragmentos_inferencia(
        df_fragmentos, tokenizer, FRAGMENT_REBALANCE_THRESHOLD
    )

    df_resultado.to_csv(fragments_path, index=False, encoding="utf-8")
    print(f"[*] Filas antes: {n_antes} -> después: {len(df_resultado)} (rechazadas: {len(df_rechazadas)})")
    print(f"[+] {fragments_path.name} sobrescrito con los fragmentos rebalanceados.")

    # Se guarda SIEMPRE, aunque quede vacío (mismo criterio que rejected_fragments_audit.csv)
    rejected_path.parent.mkdir(parents=True, exist_ok=True)
    df_rechazadas.to_csv(rejected_path, index=False, encoding="utf-8")
    print(f"[+] Rechazos guardados en: {rejected_path.name} ({len(df_rechazadas)} filas)")

    resumen = resumir_inferencia(df_resultado, df_rechazadas, FRAGMENT_REBALANCE_THRESHOLD, model_type)
    print("\n" + resumen)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(resumen, encoding="utf-8")
    print(f"[+] Reporte guardado en: {report_path.name}")

    print("[INFO] Rebalanceo de fragmentos completado.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebalancea fragmentos de inferencia largos antes de la inferencia neuronal.")
    parser.add_argument("--model", type=str, choices=["baseline", "balanced"], default="baseline")
    args = parser.parse_args()
    main(args.model)
