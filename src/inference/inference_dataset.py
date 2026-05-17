"""
src/inference/inference_dataset.py

Dataset especializado para el proceso de Inferencia Enriquecida de contratos.
Hereda de torch.utils.data.Dataset para integrarse nativamente con DataLoaders.

A diferencia de 'ContractDataset', esta clase está diseñada para "datos ciegos":
1. NO busca ni exige la columna 'categoria' (Labels), ya que el objetivo es predecirlas.
2. Está acoplada a la columna 'contenido' generada por el script de fragmentación.
"""

import torch
from torch.utils.data import Dataset
from src.config.text_standards import MAX_TOKENS

class BERTInferenceDataset(Dataset):
    """
    Encapsula una lista de fragmentos de texto crudo y los transforma 
    en tensores numéricos asimilables por el modelo BERT en modo evaluación.
    """

    def __init__(self, texts, tokenizer, max_length=MAX_TOKENS):
        """
        Inicializa el lector de datos para inferencia.

        Parámetros
        ----------
        texts : list
            Lista de strings que contiene el texto de los fragmentos (columna 'contenido').
        tokenizer : AutoTokenizer
            Tokenizador oficial de BERT cargado desde el directorio del modelo final.
        max_length : int, opcional
            Límite físico de tokens (por defecto usa MAX_TOKENS de text_standards, ej: 512).
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