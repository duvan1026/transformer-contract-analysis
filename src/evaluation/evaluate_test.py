import torch
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Añadir raíz del proyecto al path
sys.path.append(str(Path(__file__).resolve().parents[2]))

# Configuración de rutas (ajusta según tu proyecto)
from src.config.paths import (
    TEST_DATASET, MODEL_BERT_FINAL_DIR, TEST_DETAILED_RESULT, 
    CONFUSION_MATRIX, TEST_METRICS_REPORT
    )

from src.training.bert_dataset import ContractDataset
from src.config.label_mapping import ID2LABEL

def run_evaluation():
    """
    Ejecuta el pipeline de evaluación final:
    1. Carga el modelo entrenado.
    2. Realiza inferencia sobre el set de test.
    3. Genera métricas de rendimiento (CSV detallado, Matriz de Confusión y Reporte TXT).
    """

    # Selección de hardware: Usar GPU (cuda) si está disponible para mayor velocidad
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluando en: {device}")

    # 1. Cargar Tokenizer y Modelo
    # Cargamos el tokenizador y el modelo desde la carpeta donde se guardó el mejor checkpoint
    tokenizer = AutoTokenizer.from_pretrained(MODEL_BERT_FINAL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_BERT_FINAL_DIR).to(device)

    # IMPORTANTE: Poner el modelo en modo evaluación para desactivar capas de Dropout
    model.eval()

    # 2. Preparar Dataset y DataLoader
    test_dataset = ContractDataset(TEST_DATASET, tokenizer=tokenizer)
    # batch_size=16 es un balance seguro entre velocidad y uso de memoria VRAM
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    all_preds = []          # Guardará los IDs de las clases predichas
    all_labels = []         # Guardará los IDs de las clases reales
    all_confidences = []    # Guardará la probabilidad (certeza) de cada predicción

    # Recuperar datos originales para que el CSV final incluya el texto original
    df_test_raw = pd.read_csv(TEST_DATASET)

    # 3. PROCESO DE INFERENCIA (TESTING)
    print("Ejecutando inferencia sobre el conjunto de test...")

    # Desactivamos el cálculo de gradientes para ahorrar memoria y acelerar el proceso
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # El modelo devuelve 'logits' (puntuaciones crudas antes de probabilidad)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Convertimos logits a probabilidades (0 a 1) usando Softmax
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # Obtenemos la confianza (probabilidad más alta) y la predicción (índice)
            conf, preds = torch.max(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(conf.cpu().numpy())


    # 4. CONSOLIDACIÓN DE RESULTADOS Y AUDITORÍA
    results_df = df_test_raw.copy()
    results_df["pred_label_id"] = all_preds
    results_df["pred_label_name"] = [ID2LABEL[p] for p in all_preds]

    # Redondeamos la confianza para legibilidad
    results_df["confidence"] = np.round(all_confidences, 4)
    results_df["confidence_pct"] = (results_df["confidence"] * 100).round(2).astype(str) + '%'

    # Comprobación de acierto: Comparamos etiquetas ignorando mayúsculas/minúsculas
    results_df["is_correct"] = results_df["categoria"].str.lower() == results_df["pred_label_name"].str.lower()

    # 4. Guardar CSV de Auditoría (Errores y Aciertos)
    results_df.to_csv(TEST_DETAILED_RESULT, index=False)
    print(f"Reporte detallado guardado en: {TEST_DETAILED_RESULT}")

    # 5. VISUALIZACIÓN: MATRIZ DE CONFUSIÓN
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=list(ID2LABEL.values()), 
                yticklabels=list(ID2LABEL.values()))
    
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix - BERT Baseline', fontsize=14, pad=20)

    # tight_layout evita que se corten las etiquetas de los ejes al guardar la imagen
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX, dpi=300)
    plt.close()
    
    # 6. REPORTE FINAL DE MÉTRICAS (TXT)
    report = classification_report(all_labels, all_preds, target_names=list(ID2LABEL.values()))
    with open(TEST_METRICS_REPORT, "w") as f:
        f.write("REPORTE DE EVALUACIÓN FINAL - ANÁLISIS CONTRACTUAL BERT\n")
        f.write("="*60 + "\n")
        f.write(f"Accuracy General: {accuracy_score(all_labels, all_preds):.4f}\n\n")
        f.write(report)
        f.write("\n" + "="*60 + "\n")
        f.write("GLOSARIO DE MÉTRICAS:\n")
        f.write("- PRECISION: ¿Qué tan confiable es el modelo cuando detecta una cláusula? (Evita falsas alarmas).\n")
        f.write("- RECALL: ¿Qué porcentaje de las cláusulas reales logramos capturar? (Evita omisiones legales).\n")
        f.write("- F1-SCORE: El balance ideal entre precisión y captura (Métrica clave de éxito).\n")
        f.write("- SUPPORT: Número total de muestras de esa categoría en el set de prueba.\n")
    
    print("\nEvaluación completada con éxito.")
    print(report)


if __name__ == "__main__":
    run_evaluation()