#!/bin/bash
# Este script se ejecuta por dentro del contenedor Docker

set -e # Detiene la ejecución si hay algún error

# Capturar el primer argumento pasado al script
EXPERIMENT=$1

# Validar que el usuario haya ingresado una opción válida
if [ "$EXPERIMENT" != "baseline" ] && [ "$EXPERIMENT" != "balanced" ]; then
    echo "================================================================"
    echo "ERROR: Debes especificar el tipo de experimento."
    echo "Uso: ./run_pipeline.sh [baseline | balanced]"
    echo "================================================================"
    exit 1
fi

echo "========================================"
echo "Iniciando Pipeline NLP - Modo: $EXPERIMENT"
echo "========================================"

echo "[0/4] Limpiando salidas de ejecuciones anteriores locales..."
rm -rf data/processed/*
rm -rf models/*
rm -rf docs/reports/*

echo "[1/4] Ejecutando preprocesamiento común..."
python src/preprocessing/extract_spans.py
python src/preprocessing/apply_category_mapping.py


# --- BIFURCACIÓN DE LÓGICA SEGÚN EL EXPERIMENTO ---

if [ "$EXPERIMENT" == "baseline" ]; then
    echo "[2/4] Ejecutando preprocesamiento y entrenamiento BASELINE..."
    python src/preprocessing/validate_dataset.py
    python src/preprocessing/split_dataset_density.py
    python src/training/train_bert_baseline.py
    
elif [ "$EXPERIMENT" == "balanced" ]; then
    echo "[2/4] Ejecutando preprocesamiento (con clase None) y entrenamiento BALANCED..."
    python src/preprocessing/generate_none_class.py
    python src/preprocessing/validate_dataset.py
    python src/preprocessing/split_dataset_density.py
    python src/training/train_bert_balanced.py
fi

# --- EVALUACIÓN E INFERENCIA EN PRODUCCIÓN ---

echo "[3/4] Evaluando el modelo y generando reportes..."
python src/evaluation/evaluate_test.py --model "$EXPERIMENT"

echo "[4/4] Ejecutando Registro Maestro y flujo de Inferencia completo..."
python src/data/00_init_master_registry.py --model "$EXPERIMENT"
python src/inference/01_prepare_batch_documents.py --model "$EXPERIMENT"
python src/inference/02_run_inference.py --model "$EXPERIMENT"
python src/inference/03_calculate_ground_truth.py --model "$EXPERIMENT"
python src/inference/04_generate_reports.py --model "$EXPERIMENT"

echo "========================================"
echo "¡Pipeline ($EXPERIMENT) finalizado con éxito!"
echo "========================================"