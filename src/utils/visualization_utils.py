"""
src/utils/visualization_utils.py

Módulo dedicado a la generación de gráficos y recursos visuales.
Aísla las dependencias pesadas como matplotlib y seaborn del resto del flujo.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix

def save_confusion_matrix_image(df: pd.DataFrame, output_path: Path, id2label: dict, title_suffix: str = ""):
    """
    Genera y guarda una Matriz de Confusión en formato PNG usando colores corporativos (Blues).
    
    Args:
        df: DataFrame que debe contener 'etiqueta_real' y 'pred_final'.
        output_path: Ruta absoluta donde se guardará la imagen.
        title_suffix: Texto dinámico para el título (ej. "Umbral 75%").
    """
    # Estandarizamos el orden de las etiquetas, poniendo NONE primero
    labels_order = ["NONE"] + [str(lbl).strip().upper() for lbl in id2label.values()]
    
    # Asegurar que no haya duplicados si el mapeo los tuviera
    labels_order = list(dict.fromkeys(labels_order))

    # Calculamos la matriz matemática
    cm = confusion_matrix(df["etiqueta_real"], df["pred_final"], labels=labels_order)
    
    # Renderizado gráfico
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="white")
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', 
        xticklabels=labels_order, yticklabels=labels_order, 
        square=True, annot_kws={"size": 11, "weight": "bold"}
    )
    
    # Estilizado
    plt.title(f"Matriz de Confusión {title_suffix}", fontsize=14, pad=20, weight='bold')
    plt.xlabel("Predicción del Modelo", fontsize=12, labelpad=15, weight='bold')
    plt.ylabel("Etiqueta Real", fontsize=12, labelpad=15, weight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    
    # Persistencia
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()