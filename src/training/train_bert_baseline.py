import sys
from pathlib import Path
import numpy as np
import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
    set_seed # Herramienta de Hugging Face para fijar semillas en el Trainer
)

# Añadir raíz del proyecto
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.training.bert_dataset import ContractDataset
from src.config.label_mapping import MODEL_NAME, NUM_LABELS_BASELINE, LABEL2ID_BASELINE
from src.config.training_config import RANDOM_SEED 

from src.config.paths import (
    TRAIN_DATASET, 
    VAL_DATASET, 
    MODEL_BERT_BASELINE_DIR
)


def set_reproducibility(seed):
    """
    Bloquea todas las fuentes de aleatoriedad para garantizar 
    que el entrenamiento sea idéntico en cada ejecución.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Configuración para que PyTorch sea determinista en GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Semilla específica para las librerías de Hugging Face
    set_seed(seed)

# Ejecutamos la fijación de semilla antes de cargar nada
set_reproducibility(RANDOM_SEED)

# 2. Tokenizer único para todos los sets
# El Tokenizer debe ser idéntico al del modelo pre-entrenado para mantener el vocabulario.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 3. Datasets 
# Instanciación de datasets usando nuestra clase personalizada ContractDataset.
train_dataset = ContractDataset(TRAIN_DATASET, tokenizer=tokenizer, label2id=LABEL2ID_BASELINE)
val_dataset = ContractDataset(VAL_DATASET, tokenizer=tokenizer, label2id=LABEL2ID_BASELINE)

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
    num_labels=NUM_LABELS_BASELINE
)

# 6. ESTRATEGIA DE ENTRENAMIENTO (HIPERPARÁMETROS)
training_args = TrainingArguments(
    output_dir=MODEL_BERT_BASELINE_DIR, # Carpeta para guardar resultados intermedios
    seed=RANDOM_SEED,                 #  SEGUNDO BLOQUEO: Semilla para el Trainer
    data_seed=RANDOM_SEED,            #  Semilla para el barajado de datos (shuffling)

    # Optimización:
    learning_rate=2e-5,             # Velocidad de aprendizaje (ajuste fino y delicado)
    per_device_train_batch_size=8,  # Cuántas muestras procesar simultáneamente en GPU/CPU
    per_device_eval_batch_size=8,
    num_train_epochs=3,              # Número de pasadas (lee datatset) completas por los datos
    weight_decay=0.01,               # Regularización para evitar que el modelo memorice (overfitting -cuando el modelo memoriza en lugar de aprender)

    # Monitoreo y Guardado:
    evaluation_strategy="epoch",     # Evaluar el rendimiento al final de cada época
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
print(f"Iniciando entrenamiento reproducible (Semilla: {RANDOM_SEED})...")
trainer.train()

# Guardado definitivo del modelo óptimo y su tokenizer asociado.
# Esto genera los archivos necesarios (.bin, .json) para realizar predicciones futuras.
trainer.save_model(MODEL_BERT_BASELINE_DIR)
tokenizer.save_pretrained(MODEL_BERT_BASELINE_DIR)