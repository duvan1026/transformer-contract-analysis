#!/bin/bash
# Este script se ejecuta por dentro del contenedor Docker

set -e # Detiene la ejecución si hay algún error

# Capturar el primer argumento pasado al script
EXPERIMENT=$1

# Validar argumento
if [ "$EXPERIMENT" != "baseline" ] && [ "$EXPERIMENT" != "balanced" ]; then
    echo "================================================================"
    echo "ERROR: Debes especificar el tipo de experimento."
    echo "Uso: ./run_pipeline.sh [baseline | balanced]"
    echo "================================================================"
    exit 1
fi

# ==============================================================================
# FUNCIÓN AUXILIAR: MIDE TIEMPO Y REGISTRA FECHA/HORA
# ==============================================================================
run_step() {
    local step_name="$1"
    shift # Elimina el primer argumento para dejar solo el comando a ejecutar
    
    local start_time=$(date +%s)
    local start_datetime=$(date '+%Y-%m-%d %H:%M:%S')

    echo "----------------------------------------------------------------"
    echo "⏱️  [$start_datetime] INICIANDO: $step_name"
    echo "----------------------------------------------------------------"
    
    # Ejecuta el comando pasado a la función
    "$@"
    
    local end_time=$(date +%s)
    local end_datetime=$(date '+%Y-%m-%d %H:%M:%S')
    local duration=$((end_time - start_time))
    local mins=$((duration / 60))
    local secs=$((duration % 60))

    echo "✅ [$end_datetime] FINALIZADO: $step_name"
    echo "⏳ Tiempo de ejecución: ${mins}m ${secs}s (Total: ${duration}s)"
    echo "----------------------------------------------------------------"
    echo ""
}

# ==============================================================================
# INICIO DEL PIPELINE
# ==============================================================================
PIPELINE_START_TIME=$(date +%s)

echo "================================================================"
echo "🚀 Iniciando Pipeline NLP - Modo: $EXPERIMENT"
echo "   Fecha y Hora de inicio: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

run_step "[0/4] Limpiando salidas locales" bash -c "rm -rf data/processed/* models/* docs/reports/* && mkdir -p data/processed models docs/reports"

run_step "[1/4] Extraer Spans" python src/preprocessing/extract_spans.py
run_step "[1/4] Aplicar Mapeo de Categorías" python src/preprocessing/apply_category_mapping.py

# --- BIFURCACIÓN DE LÓGICA SEGÚN EL EXPERIMENTO ---

if [ "$EXPERIMENT" == "baseline" ]; then
    run_step "[2/4] Validar Dataset" python src/preprocessing/validate_dataset.py
    run_step "[2/4] Dividir Dataset por Densidad" python src/preprocessing/split_dataset_density.py
    run_step "[2/4] Entrenar BERT Baseline" python src/training/train_bert_baseline.py
    
elif [ "$EXPERIMENT" == "balanced" ]; then
    run_step "[2/4] Generar Clase NONE" python src/preprocessing/generate_none_class.py
    run_step "[2/4] Validar Dataset" python src/preprocessing/validate_dataset.py
    run_step "[2/4] Dividir Dataset por Densidad" python src/preprocessing/split_dataset_density.py
    run_step "[2/4] Entrenar BERT Balanced" python src/training/train_bert_balanced.py
fi

# --- EVALUACIÓN E INFERENCIA ---

run_step "[3/4] Evaluando Modelo (Test)" python src/evaluation/evaluate_test.py --model "$EXPERIMENT"

run_step "[4/4] Inicializar Registro Maestro" python src/data/00_init_master_registry.py --model "$EXPERIMENT"
run_step "[4/4] Preparar Lote de Documentos" python src/inference/01_prepare_batch_documents.py --model "$EXPERIMENT"
run_step "[4/4] Inferencia Pura (Softmax)" python src/inference/02_run_inference.py --model "$EXPERIMENT"
run_step "[4/4] Calcular Ground Truth Espacial" python src/inference/03_calculate_ground_truth.py --model "$EXPERIMENT"
run_step "[4/4] Generar Reportes y JSONs" python src/inference/04_generate_reports.py --model "$EXPERIMENT"

# --- RESUMEN FINAL ---
PIPELINE_END_TIME=$(date +%s)
TOTAL_DURATION=$((PIPELINE_END_TIME - PIPELINE_START_TIME))
TOTAL_MINS=$((TOTAL_DURATION / 60))
TOTAL_SECS=$((TOTAL_DURATION % 60))

echo "================================================================"
echo "🎉 ¡Pipeline ($EXPERIMENT) finalizado con éxito!"
echo "   Hora de fin: $(date '+%Y-%m-%d %H:%M:%S')"
echo "⏱️  TIEMPO TOTAL DEL PIPELINE: ${TOTAL_MINS}m ${TOTAL_SECS}s (${TOTAL_DURATION}s)"
echo "================================================================"