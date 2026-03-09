import torch
import sys
from pathlib import Path
from torch.utils.data import Dataset
import pandas as pd

# Añadir raíz del proyecto
sys.path.append(str(Path(__file__).resolve().parents[2]))

from transformers import AutoTokenizer
from src.config.label_mapping import LABEL2ID


class ContractDataset(Dataset):
    """
    Clase que actúa como interfaz entre los archivos CSV de contratos y los modelos de Hugging Face.
    Hereda de torch.utils.data.Dataset para integrarse con DataLoaders de PyTorch.
    """

    def __init__(self, file_path, tokenizer, max_length=512):
        """
        Inicializa el lector de datos.
        
        Args:
            file_path (str): Ruta al archivo CSV con los datos de entrenamiento/evaluación.
            tokenizer (AutoTokenizer): Tokenizer de BERT cargado para procesar el texto.
            max_length (int): Límite de tokens por secuencia (BERT soporta hasta 512).
        """

        self.data = pd.read_csv(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length


    def __len__(self):
        """Devuelve el número total de muestras en el dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Recupera y procesa una sola muestra del dataset dado un índice.

        Args:
            idx (int o tensor): Índice del elemento que se desea recuperar. 
                                El DataLoader de PyTorch lo gestiona automáticamente 
                                para crear 'batches' (lotes) de datos.
        
        Proceso:
        1. Limpieza y normalización de texto y etiquetas.
        2. Tokenización (conversión de texto a IDs numéricos).
        3. Empaquetado en tensores de PyTorch.
        """
        # Extraer fila por índice
        row = self.data.iloc[idx]

        # Aseguramos que el texto sea string (evita errores con NaNs)
        text = str(row["span_text"])

        # Mapeo semántico: Convertir nombre de categoría (ej. 'Audit Rights') a ID numérico (ej. 0)
        # Se normaliza a minúsculas y se eliminan espacios para evitar desajustes en el diccionario.
        label_name = row["categoria"].lower().strip()
        label = LABEL2ID[label_name]

        # Proceso de Tokenización:
        # - padding: Rellena secuencias cortas con tokens [PAD] hasta llegar a max_length.
        # - truncation: Corta secuencias que excedan el max_length.
        # - return_token_type_ids: Falso porque no estamos usando pares de frases (Pregunta/Respuesta).
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            # No pedimos tensores aquí para manipularlo más fácil
            return_token_type_ids=False
        )

        # Retornar diccionario de tensores listos para ser consumidos por BERT
        # input_ids: La representación numérica de los tokens.
        # attention_mask: Binario (1 para texto real, 0 para relleno [PAD]).
        # labels: El ID de la categoría (Ground Truth).
        return {
            "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long)
        }