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
from src.config.paths import (
    TRAIN_DATASET, 
    VAL_DATASET, 
    MODEL_BERT_BASELINE_DIR,
    MODEL_BERT_FINAL_DIR
)

# 1. Configuración Centralizada
MODEL_NAME = "bert-base-uncased"
NUM_LABELS = 5

# 2. Tokenizer único para todos los sets
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 3. Datasets (Pasando el tokenizer instanciado)
train_dataset = ContractDataset(TRAIN_DATASET, tokenizer=tokenizer)
val_dataset = ContractDataset(VAL_DATASET, tokenizer=tokenizer)

# 4. Definir métricas (Crítico para NLP Legal)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    # Usamos 'macro' porque CUAD suele tener clases desbalanceadas
    f1 = f1_score(labels, predictions, average="macro")
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1}

# 5. Modelo
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS
)

# 6. Argumentos de Entrenamiento Optimizados
training_args = TrainingArguments(
    output_dir=MODEL_BERT_BASELINE_DIR,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,               # Ayuda a prevenir overfitting

    eval_strategy="epoch",
    save_strategy="epoch",           # Guardar checkpoint cada época
    logging_dir="logs",
    logging_steps=10,
    load_best_model_at_end=True,     # Al final, quédate con el que mejor F1 tuvo
    metric_for_best_model="f1",
)

# 7. Trainer Completo
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Ejecución
print("Iniciando entrenamiento...")
trainer.train()

# Guardar el modelo final y el tokenizer (importante para inferencia posterior)
trainer.save_model(MODEL_BERT_FINAL_DIR)
tokenizer.save_pretrained(MODEL_BERT_FINAL_DIR)