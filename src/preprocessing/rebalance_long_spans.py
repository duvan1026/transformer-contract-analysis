"""
src/preprocessing/rebalance_long_spans.py

Aplica el rebalanceo de spans largos (src/utils/rebalancing/) a train.csv,
val.csv y test.csv, sobrescribiendo cada archivo en su misma ruta. Todas
las filas se procesan por igual, sean de una clase positiva o de NONE.

Se ejecuta DESPUÉS de split_dataset_density.py / split_dataset_stratified.py
y ANTES de train_bert_baseline.py / train_bert_balanced.py -- después de
que la selección de contratos/documentos por split ya quedó fija, para no
alterar esa selección (ver Analisis... y discusión de arquitectura: dividir
antes del split cambiaría n_spans/n_categorias por documento y por tanto
qué contratos se eligen).

No requiere --model: TRAIN_DATASET/VAL_DATASET/TEST_DATASET son las mismas
rutas físicas en ambas ramas del pipeline; el contenido depende de qué
split_dataset_*.py corrió antes, no de este script (mismo criterio que ya
usa validate_dataset.py).
"""

import sys
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config.paths import (
    TRAIN_DATASET,
    VAL_DATASET,
    TEST_DATASET,
    REBALANCE_REJECTED_TRAIN,
    REBALANCE_REPORT_TRAIN,
)
from src.config.text_standards import FRAGMENT_REBALANCE_THRESHOLD, MODEL_NAME
from src.utils.rebalancing.io import rebalancear_split_train
from src.utils.rebalancing.stats import resumir_train

_SPLITS = [
    ("train", TRAIN_DATASET),
    ("val", VAL_DATASET),
    ("test", TEST_DATASET),
]


def run():
    print(f"[*] Cargando tokenizador: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    reportes = []
    rechazados_por_split = []

    for nombre_split, ruta in _SPLITS:
        if not ruta.exists():
            print(f"[AVISO] No se encontró {ruta} -- se omite '{nombre_split}'.")
            continue

        print(f"\n[*] Rebalanceando '{nombre_split}' ({ruta.name})...")
        df_split = pd.read_csv(ruta)
        n_antes = len(df_split)

        df_resultado, df_rechazadas = rebalancear_split_train(
            df_split, tokenizer, FRAGMENT_REBALANCE_THRESHOLD
        )

        df_resultado.to_csv(ruta, index=False, encoding="utf-8")
        print(f"    Filas antes: {n_antes} -> después: {len(df_resultado)} "
              f"(rechazadas: {len(df_rechazadas)})")

        if not df_rechazadas.empty:
            df_rechazadas = df_rechazadas.copy()
            df_rechazadas.insert(0, "split", nombre_split)
            rechazados_por_split.append(df_rechazadas)

        resumen = resumir_train(df_resultado, df_rechazadas, FRAGMENT_REBALANCE_THRESHOLD, nombre_split)
        reportes.append(resumen)
        print(resumen)

    # El CSV de rechazos se guarda SIEMPRE, aunque quede vacío (mismo
    # criterio que rejected_fragments_audit.csv en 01_prepare_batch_documents.py)
    REBALANCE_REJECTED_TRAIN.parent.mkdir(parents=True, exist_ok=True)
    if rechazados_por_split:
        df_rechazados_final = pd.concat(rechazados_por_split, ignore_index=True)
    else:
        df_rechazados_final = pd.DataFrame(columns=["split"])
    df_rechazados_final.to_csv(REBALANCE_REJECTED_TRAIN, index=False, encoding="utf-8")
    print(f"\n[+] Rechazos guardados en: {REBALANCE_REJECTED_TRAIN.name} ({len(df_rechazados_final)} filas)")

    REBALANCE_REPORT_TRAIN.parent.mkdir(parents=True, exist_ok=True)
    REBALANCE_REPORT_TRAIN.write_text("\n\n".join(reportes), encoding="utf-8")
    print(f"[+] Reporte guardado en: {REBALANCE_REPORT_TRAIN.name}")


if __name__ == "__main__":
    run()
