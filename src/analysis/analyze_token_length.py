"""
analyze_token_length.py

Script orquestador de un analisis DIAGNOSTICO (no forma parte del pipeline
obligatorio de run_pipeline.sh - se ejecuta manualmente cuando se necesite).

Objetivo:
Comparar la longitud (en tokens BERT y en palabras) de los fragmentos de
texto por clase entre:
  - Entrenamiento (train / val / test): span_text, spans CUAD exactos.
  - Inferencia real (ground truth): contenido, fragmentos por parrafo.

Toda la logica de carga, calculo, formato de Excel y graficado vive en
src/utils/ (token_length_io, token_length_stats, token_length_excel_writer,
token_length_plots, tokenization_utils). Este script solo orquesta: CLI,
enrutamiento por --model, flujo y guardado.

Los umbrales de referencia (TOKEN_LENGTH_WORD_THRESHOLD,
TOKEN_LENGTH_RISK_ZONE_TOKENS) viven en text_standards.py, no estan
hardcodeados aqui ni en paths.py — si cambian, el analisis, el Excel y las
graficas se actualizan automaticamente en la siguiente corrida.

Uso:
    python src/analysis/analyze_token_length.py --model baseline
    python src/analysis/analyze_token_length.py --model balanced

Requiere que train.csv / val.csv / test.csv en disco correspondan al split
generado por el MISMO experimento indicado en --model.
"""

import sys
import json
import argparse
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

# Anadir raiz del proyecto
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config.paths import (
    TRAIN_DATASET,
    VAL_DATASET,
    TEST_DATASET,
    GROUND_TRUTH_BASELINE,
    GROUND_TRUTH_BALANCED,
    TOKEN_LENGTH_BASELINE_DIR,
    TOKEN_LENGTH_STATS_BASELINE,
    TOKEN_LENGTH_BOXPLOT_BASELINE,
    TOKEN_LENGTH_ECDF_BASELINE,
    TOKEN_LENGTH_LABEL_AUDIT_BASELINE,
    TOKEN_LENGTH_CLASS_BALANCE_BASELINE,
    TOKEN_LENGTH_COMPOSITION_WITH_NONE_BASELINE,
    TOKEN_LENGTH_COMPOSITION_POSITIVE_ONLY_BASELINE,
    TOKEN_LENGTH_TRUNCATION_RISK_BASELINE,
    TOKEN_LENGTH_RISK_BY_CLASS_BASELINE,
    TOKEN_LENGTH_BALANCED_DIR,
    TOKEN_LENGTH_STATS_BALANCED,
    TOKEN_LENGTH_BOXPLOT_BALANCED,
    TOKEN_LENGTH_ECDF_BALANCED,
    TOKEN_LENGTH_LABEL_AUDIT_BALANCED,
    TOKEN_LENGTH_CLASS_BALANCE_BALANCED,
    TOKEN_LENGTH_COMPOSITION_WITH_NONE_BALANCED,
    TOKEN_LENGTH_COMPOSITION_POSITIVE_ONLY_BALANCED,
    TOKEN_LENGTH_TRUNCATION_RISK_BALANCED,
    TOKEN_LENGTH_RISK_BY_CLASS_BALANCED,
    RECALL_BY_LENGTH_RISK_ZONE_FILE_BASELINE,
    RECALL_BY_LENGTH_RISK_ZONE_FILE_BALANCED,
)
from src.config.text_standards import (
    MODEL_NAME,
    MAX_CONTENT_TOKENS,
    TOKEN_LENGTH_WORD_THRESHOLD,
    TOKEN_LENGTH_RISK_ZONE_TOKENS,
)
from src.config.label_mapping import LABEL2ID_BASELINE, LABEL2ID_BALANCED

from src.utils.token_length.io import cargar_split, cargar_inferencia_ground_truth
from src.utils.token_length.stats import (
    calcular_estadisticos,
    construir_resumen_comparativo,
    construir_tabla_conteos,
    construir_tabla_composicion,
    construir_tabla_riesgo_truncamiento,
    construir_tabla_riesgo_por_clase_inferencia,
)
from src.utils.token_length.excel_writer import guardar_excel_estadisticos
from src.utils.token_length.plots import (
    graficar_boxplot,
    graficar_ecdf,
    graficar_balance_clases,
    graficar_composicion_relativa,
    graficar_riesgo_truncamiento,
    graficar_riesgo_por_clase,
)


def main(model_type):
    print(f"Iniciando analisis de longitud de tokens ({model_type.upper()})...")
    print("[AVISO] Este analisis asume que train.csv/val.csv/test.csv en disco "
          f"corresponden al split generado por la ultima ejecucion de "
          f"run_pipeline.sh {model_type}. Si el pipeline se corrio mas "
          f"recientemente con el otro experimento, los resultados de train/val/test "
          f"no seran representativos de este modo.\n")

    # 1. Enrutar segun el modelo seleccionado
    if model_type == "balanced":
        ground_truth_path = GROUND_TRUTH_BALANCED
        etiquetas_validas = set(LABEL2ID_BALANCED.keys())
        output_dir = TOKEN_LENGTH_BALANCED_DIR
        stats_path = TOKEN_LENGTH_STATS_BALANCED
        boxplot_path = TOKEN_LENGTH_BOXPLOT_BALANCED
        ecdf_path = TOKEN_LENGTH_ECDF_BALANCED
        label_audit_path = TOKEN_LENGTH_LABEL_AUDIT_BALANCED
        class_balance_path = TOKEN_LENGTH_CLASS_BALANCE_BALANCED
        composition_with_none_path = TOKEN_LENGTH_COMPOSITION_WITH_NONE_BALANCED
        composition_positive_only_path = TOKEN_LENGTH_COMPOSITION_POSITIVE_ONLY_BALANCED
        truncation_risk_path = TOKEN_LENGTH_TRUNCATION_RISK_BALANCED
        risk_by_class_path = TOKEN_LENGTH_RISK_BY_CLASS_BALANCED
        risk_zone_file_path = RECALL_BY_LENGTH_RISK_ZONE_FILE_BALANCED
    else:
        ground_truth_path = GROUND_TRUTH_BASELINE
        # Baseline nunca entreno la clase NONE -> se excluye del set de etiquetas validas
        etiquetas_validas = set(LABEL2ID_BASELINE.keys())
        output_dir = TOKEN_LENGTH_BASELINE_DIR
        stats_path = TOKEN_LENGTH_STATS_BASELINE
        boxplot_path = TOKEN_LENGTH_BOXPLOT_BASELINE
        ecdf_path = TOKEN_LENGTH_ECDF_BASELINE
        label_audit_path = TOKEN_LENGTH_LABEL_AUDIT_BASELINE
        class_balance_path = TOKEN_LENGTH_CLASS_BALANCE_BASELINE
        composition_with_none_path = TOKEN_LENGTH_COMPOSITION_WITH_NONE_BASELINE
        composition_positive_only_path = TOKEN_LENGTH_COMPOSITION_POSITIVE_ONLY_BASELINE
        truncation_risk_path = TOKEN_LENGTH_TRUNCATION_RISK_BASELINE
        risk_by_class_path = TOKEN_LENGTH_RISK_BY_CLASS_BASELINE
        risk_zone_file_path = RECALL_BY_LENGTH_RISK_ZONE_FILE_BASELINE

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1b. Zona de riesgo: usar la calculada por analyze_recall_by_length.py si existe,
    # o el valor de respaldo de text_standards.py con advertencia si no se ha ejecutado.
    zona_riesgo_tokens = TOKEN_LENGTH_RISK_ZONE_TOKENS
    if risk_zone_file_path.exists():
        with open(risk_zone_file_path, "r", encoding="utf-8") as f:
            zona_riesgo_calculada = json.load(f)
        zona_riesgo_tokens = (zona_riesgo_calculada["bin_min"], zona_riesgo_calculada["bin_max"])
        print(f"Usando zona de riesgo calculada por analyze_recall_by_length.py: "
              f"{zona_riesgo_tokens[0]}-{zona_riesgo_tokens[1]} tokens "
              f"(recall {zona_riesgo_calculada['recall']}%, n={zona_riesgo_calculada['n']}).\n")
    else:
        print(f"[AVISO] No se encontro {risk_zone_file_path}. Usando el valor de respaldo "
              f"{zona_riesgo_tokens[0]}-{zona_riesgo_tokens[1]} tokens de text_standards.py. "
              f"Para una zona de riesgo verificada con datos actuales, ejecuta primero:\n"
              f"  python src/analysis/analyze_recall_by_length.py --model {model_type}\n")

    # 2. Tokenizador (mismo patron que 01_prepare_batch_documents.py: sin checkpoint local)
    print(f"Cargando tokenizador: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    registro_no_reconocidas = []

    # 3. Cargar y medir cada fuente
    print("Procesando train...")
    df_train = cargar_split(
        TRAIN_DATASET, "train", "span_text", "categoria",
        etiquetas_validas, tokenizer, registro_no_reconocidas
    )
    print("Procesando val...")
    df_val = cargar_split(
        VAL_DATASET, "val", "span_text", "categoria",
        etiquetas_validas, tokenizer, registro_no_reconocidas
    )
    print("Procesando test...")
    df_test = cargar_split(
        TEST_DATASET, "test", "span_text", "categoria",
        etiquetas_validas, tokenizer, registro_no_reconocidas
    )
    print("Procesando inferencia (ground truth)...")
    if model_type == "baseline" and not ground_truth_path.exists():
        print(f"  [AVISO] {ground_truth_path} no existe todavia para el carril baseline. "
              f"Se omite el lado de inferencia; solo se reportara train/val/test.")
        df_inferencia = pd.DataFrame(columns=["fuente", "clase", "n_palabras", "n_tokens"])
    else:
        df_inferencia = cargar_inferencia_ground_truth(
            ground_truth_path, etiquetas_validas, registro_no_reconocidas
        )

    df_todas_fuentes = pd.concat(
        [df_train, df_val, df_test, df_inferencia], ignore_index=True
    )

    if df_todas_fuentes.empty:
        print("[ERROR] No se pudo cargar ninguna fuente con datos validos. Abortando.")
        return

    # 4. Estadisticos por fuente (umbrales tomados de text_standards.py, no hardcodeados)
    stats_por_fuente = {}
    for fuente in ["train", "val", "test", "inferencia"]:
        sub = df_todas_fuentes[df_todas_fuentes["fuente"] == fuente]
        stats_por_fuente[fuente] = calcular_estadisticos(
            sub, TOKEN_LENGTH_WORD_THRESHOLD, zona_riesgo_tokens, MAX_CONTENT_TOKENS
        )

    resumen_comparativo = construir_resumen_comparativo(stats_por_fuente)

    # 5. Guardar Excel formateado (diccionario de variables + una hoja por fuente + resumen)
    print(f"Guardando estadisticos en {stats_path}")
    guardar_excel_estadisticos(
        stats_por_fuente, resumen_comparativo, stats_path,
        umbral_palabras=TOKEN_LENGTH_WORD_THRESHOLD,
        zona_riesgo_tokens=zona_riesgo_tokens,
        max_tokens=MAX_CONTENT_TOKENS,
    )

    # 6. Graficas (sin IDs de hallazgo, etiqueta descriptiva dinamica)
    print(f"Generando boxplot en {boxplot_path}")
    graficar_boxplot(df_todas_fuentes, boxplot_path, title_suffix=f"({model_type})")

    print(f"Generando ECDF en {ecdf_path}")
    graficar_ecdf(
        df_todas_fuentes, ecdf_path, title_suffix=f"({model_type})",
        umbral_palabras=TOKEN_LENGTH_WORD_THRESHOLD,
        zona_riesgo_tokens=zona_riesgo_tokens,
    )

    # 6b. Balance de clases, composicion relativa y riesgo de truncamiento
    df_conteos = construir_tabla_conteos(stats_por_fuente)
    print(f"Generando balance de clases en {class_balance_path}")
    graficar_balance_clases(df_conteos, class_balance_path, title_suffix=f"({model_type})")

    df_composicion_con_none = construir_tabla_composicion(stats_por_fuente, incluir_none=True)
    print(f"Generando composicion relativa (con NONE) en {composition_with_none_path}")
    graficar_composicion_relativa(
        df_composicion_con_none, composition_with_none_path,
        title_suffix=f"({model_type})", incluir_none=True
    )

    df_composicion_sin_none = construir_tabla_composicion(stats_por_fuente, incluir_none=False)
    print(f"Generando composicion relativa (solo positivas) en {composition_positive_only_path}")
    graficar_composicion_relativa(
        df_composicion_sin_none, composition_positive_only_path,
        title_suffix=f"({model_type})", incluir_none=False
    )

    df_riesgo_truncamiento = construir_tabla_riesgo_truncamiento(stats_por_fuente)
    print(f"Generando riesgo de truncamiento en {truncation_risk_path}")
    graficar_riesgo_truncamiento(
        df_riesgo_truncamiento, truncation_risk_path, title_suffix=f"({model_type})"
    )

    df_riesgo_por_clase = construir_tabla_riesgo_por_clase_inferencia(stats_por_fuente)
    print(f"Generando ranking de riesgo por clase en {risk_by_class_path}")
    graficar_riesgo_por_clase(
        df_riesgo_por_clase, risk_by_class_path, title_suffix=f"({model_type})"
    )

    # 7. Auditoria de etiquetas no reconocidas
    if registro_no_reconocidas:
        df_auditoria = pd.concat(registro_no_reconocidas, ignore_index=True)
        df_auditoria.to_csv(label_audit_path, index=False, encoding="utf-8")
        print(f"[AVISO] Se registraron etiquetas no reconocidas en {label_audit_path}")
    else:
        print("Sin etiquetas no reconocidas. No se genera archivo de auditoria.")

    print(f"\nAnalisis de longitud de tokens ({model_type}) finalizado.")
    print(f"Resultados en: {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analiza la longitud (tokens/palabras) por clase en train/val/test vs. inferencia real."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["baseline", "balanced"],
        required=True,
        help="Selecciona el carril de inferencia a comparar: 'baseline' o 'balanced'.",
    )
    args = parser.parse_args()
    main(args.model)