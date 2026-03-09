import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments
)

# Añadir raíz del proyecto
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.training.bert_dataset import ContractDataset
from src.config.label_mapping import MODEL_NAME, NUM_LABELS
from src.config.paths import (
    TRAIN_DATASET, 
    VAL_DATASET, 
    MODEL_BERT_BASELINE_DIR,
    MODEL_BERT_FINAL_DIR
)

# 1. Configuración Centralizada
MODEL_NAME = MODEL_NAME          # Usamos bert-base-uncased: 110M de parámetros, arquitectura balanceada para clasificación.
NUM_LABELS = NUM_LABELS          # Define la dimensión de la capa de salida (una neurona por categoría legal).

# 2. Tokenizer único para todos los sets
# El Tokenizer debe ser idéntico al del modelo pre-entrenado para mantener el vocabulario.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 3. Datasets 
# Instanciación de datasets usando nuestra clase personalizada ContractDataset.
train_dataset = ContractDataset(TRAIN_DATASET, tokenizer=tokenizer)
val_dataset = ContractDataset(VAL_DATASET, tokenizer=tokenizer)

# 4. Definir métricas (Crítico para NLP Legal)
# Usamos Macro F1 para compensar el desbalance de clases legal
def compute_metrics(eval_pred):
    """
    Calcula el rendimiento del modelo durante la validación.
    
    Logits: Valores crudos de la última capa antes de la activación.
    Argmax: Selecciona el índice (clase) con el valor de logit más alto.
    F1-Macro: Promedia el F1 de cada clase tratándolas con igual importancia, 
              crucial para detectar si el modelo ignora clases minoritarias.
    """

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Usamos 'macro' porque CUAD suele tener clases desbalanceadas
    f1 = f1_score(labels, predictions, average="macro")
    acc = accuracy_score(labels, predictions)

    return {"accuracy": acc, "f1": f1}



# 5. Modelo
# Se carga BERT
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS
)

# 6. ESTRATEGIA DE ENTRENAMIENTO (HIPERPARÁMETROS)
training_args = TrainingArguments(
    output_dir=MODEL_BERT_BASELINE_DIR, # Carpeta para guardar resultados intermedios

    # Optimización:
    learning_rate=2e-5,             # Velocidad de aprendizaje (ajuste fino y delicado)
    per_device_train_batch_size=8,  # Cuántas muestras procesar simultáneamente en GPU/CPU
    per_device_eval_batch_size=8,
    num_train_epochs=3,              # Número de pasadas (lee datatset) completas por los datos
    weight_decay=0.01,               # Regularización para evitar que el modelo memorice (overfitting -cuando el modelo memoriza en lugar de aprender)

    # Monitoreo y Guardado:
    eval_strategy="epoch",           # Evaluar el rendimiento al final de cada época
    save_strategy="epoch",           # Guardar un punto de control (checkpoint) cada época
    logging_dir="logs",
    logging_steps=10,

    # Selección del mejor modelo:
    load_best_model_at_end=True,     # Garantiza que el modelo final sea el más óptimo Al final
    metric_for_best_model="f1",      # El criterio principal de éxito es el F1-Macro
)

# 7. ORQUESTADOR (TRAINER)
# El Trainer gestiona el bucle de entrenamiento, optimización y evaluación.
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# EJECUCIÓN Y PERSISTENCIA
print("Iniciando entrenamiento...")
trainer.train()

# Guardado definitivo del modelo óptimo y su tokenizer asociado.
# Esto genera los archivos necesarios (.bin, .json) para realizar predicciones futuras.
trainer.save_model(MODEL_BERT_FINAL_DIR)
tokenizer.save_pretrained(MODEL_BERT_FINAL_DIR)