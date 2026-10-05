"""
src/inference/inference_dataset.py

Dataset especializado para el proceso de Inferencia Enriquecida de contratos.
Hereda de torch.utils.data.Dataset para integrarse nativamente con DataLoaders.

Dataset utilizado durante la inferencia, cuando las etiquetas reales
no forman parte de la entrada.
"""

import torch
from torch.utils.data import Dataset
from src.config.text_standards import MAX_MODEL_TOKENS

class BERTInferenceDataset(Dataset):
    """
    Encapsula una lista de fragmentos de texto crudo y los transforma 
    en tensores numéricos asimilables por el modelo BERT en modo evaluación.
    """

    def __init__(self, texts, tokenizer, max_length=MAX_MODEL_TOKENS):
        """
        Inicializa el lector de datos para inferencia.

        Parámetros
        ----------
        texts : list
            Lista de strings que contiene el texto de los fragmentos (columna 'contenido').
        tokenizer : AutoTokenizer
            Tokenizador oficial de BERT cargado desde el directorio del modelo final.
        max_length : int, opcional
            Límite físico de tokens, con [CLS]/[SEP] incluidos (por defecto usa MAX_MODEL_TOKENS de text_standards, 512).
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """Devuelve el volumen total de fragmentos a procesar en el batch."""
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Recupera un fragmento, lo tokeniza y lo prepara en formato tensor.

        Parámetros
        ----------
        idx : int
            Índice del fragmento solicitado por el DataLoader.

        Retorna
        -------
        dict
            Diccionario con 'input_ids' y 'attention_mask' listos para la GPU/CPU.
        """
        # Forzar que el texto sea un string limpio para evitar errores si Pandas lee un NaN
        text = str(self.texts[idx]).strip()

        # Proceso de Tokenización de Inferencia (Exactamente igual al de entrenamiento)
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_token_type_ids=False,  # Falso: Clasificación de texto único, no pares (Q&A)
            return_tensors="pt"           # Retorna tensores de PyTorch directamente
        )

        # .squeeze(0) es crítico aquí porque 'return_tensors="pt"' añade una dimensión extra 
        # de tamaño 1 (ej: [1, 512]). El DataLoader necesita que sea plano ([512]) para 
        # poder agruparlos correctamente en lotes (batches).
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0)
        }