import sys
from pathlib import Path
import numpy as np
import random
import torch
from torch import nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
    set_seed
)

# Añadir raíz del proyecto
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.training.bert_dataset import ContractDataset
from src.config.label_mapping import MODEL_NAME, NUM_LABELS_BALANCED, LABEL2ID_BALANCED
from src.config.training_config import RANDOM_SEED 

from src.config.paths import (
    TRAIN_DATASET, 
    VAL_DATASET, 
    MODEL_BERT_BALANCED_DIR
)

def set_reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(seed)

set_reproducibility(RANDOM_SEED)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_dataset = ContractDataset(TRAIN_DATASET, tokenizer=tokenizer, label2id=LABEL2ID_BALANCED)
val_dataset = ContractDataset(VAL_DATASET, tokenizer=tokenizer, label2id=LABEL2ID_BALANCED)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, predictions, average="macro")
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1}

# --- NUEVO: CÁLCULO DE PESOS DE CLASE ---
print("Calculando pesos de clase para mitigar el desbalance...")
train_labels = [train_dataset[i]["labels"].item() for i in range(len(train_dataset))]
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels),
    y=train_labels
)
print(f"Pesos calculados: {class_weights}")

#  TRAINER PERSONALIZADO ---
class BalancedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Movemos los pesos al mismo dispositivo que el modelo (CPU o GPU)
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Inyectamos los pesos en la función de pérdida
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS_BALANCED
)

training_args = TrainingArguments(
    output_dir=MODEL_BERT_BALANCED_DIR, 
    seed=RANDOM_SEED,                 
    data_seed=RANDOM_SEED,            
    learning_rate=2e-5,             
    per_device_train_batch_size=8,  
    per_device_eval_batch_size=8,
    num_train_epochs=3,              
    weight_decay=0.01,               
    evaluation_strategy="epoch",     
    save_strategy="epoch",           
    logging_dir="logs_balanced",
    logging_steps=10,
    load_best_model_at_end=True,     
    metric_for_best_model="f1",      
)

# Usamos nuestro BalancedTrainer en lugar del Trainer estándar
trainer = BalancedTrainer(
    class_weights=class_weights,
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

print(f"Iniciando entrenamiento balanceado reproducible (Semilla: {RANDOM_SEED})...")
trainer.train()

trainer.save_model(MODEL_BERT_BALANCED_DIR)
tokenizer.save_pretrained(MODEL_BERT_BALANCED_DIR)