#!/bin/bash
# Este script se ejecuta por dentro del contenedor Docker

echo "1. Ejecutando preprocesamiento..."
python src/preprocessing/extract_spans.py
python src/preprocessing/apply_category_mapping.py
python src/preprocessing/validate_dataset.py
python src/preprocessing/split_dataset.py

echo "2. Iniciando entrenamiento de BERT..."
python src/training/train_bert.py

echo "3. Evaluando el modelo y generando reportes..."
python src/evaluation/evaluate_test.py

echo "¡Pipeline finalizado con éxito!"