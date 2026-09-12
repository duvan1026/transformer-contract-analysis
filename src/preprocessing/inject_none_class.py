"""
inject_none_class.py

Script responsable de inyectar la clase NONE en el dataset principal.
Aplica una lógica de balanceo local (por documento):
1. Calcula el volumen máximo de spans de cualquier categoría positiva dentro de un contrato.
2. Muestrea aleatoriamente esa misma cantidad exacta de fragmentos NONE del archivo maestro.
3. Concatena los resultados manteniendo la semilla determinista para asegurar la reproducibilidad.
"""

import sys
import pandas as pd
from pathlib import Path

# Añadir raíz del proyecto al path
sys.path.append(str(Path(__file__).resolve().parents[2]))

# Importar rutas y configuraciones centralizadas
from src.config.paths import (
    DATASET_TAXONOMY_CSV,
    DATASET_NONE_MASTER_CSV
)
from src.config.training_config import RANDOM_SEED


def run():
    print("[INFO] Iniciando inyección y balanceo local de la clase NONE...")

    # 1. Cargar datasets
    try:
        df_taxonomy = pd.read_csv(DATASET_TAXONOMY_CSV)
        df_master_none = pd.read_csv(DATASET_NONE_MASTER_CSV)
    except FileNotFoundError as e:
        print(f"[ERROR] No se encontraron los archivos base: {e}")
        sys.exit(1)

    # 2. Preparar columnas del master NONE para que coincidan con taxonomy
    # Aseguramos de renombrar 'start_offset' a 'offset' si viene así del generador
    if "start_offset" in df_master_none.columns:
        df_master_none = df_master_none.rename(columns={"start_offset": "offset"})

    if "end_offset" in df_master_none.columns:
        df_master_none = df_master_none.rename(columns={"end_offset": "offset_end"})

    # offset_end ya viene calculado por generate_none_class.py sobre el texto
    # crudo (match.end()) — se conserva tal cual. NO se recalcula aquí a partir
    # de offset + len(span_text), porque span_text puede diferir en longitud
    # del texto crudo si en algún momento se introduce normalización
    # (clean_text) sobre los candidatos NONE.
    columnas_requeridas = ["id_documento", "categoria", "span_text", "offset", "offset_end"]
    columnas_existentes = [col for col in columnas_requeridas if col in df_master_none.columns]
    df_master_none = df_master_none[columnas_existentes]

    if "offset_end" not in df_master_none.columns:
        raise ValueError(
            "dataset_none_master.csv no trae 'end_offset'/'offset_end'. "
            "Verifica que generate_none_class.py esté generando esta columna "
            "antes de ejecutar inject_none_class.py."
        )       

    # 3. Lógica de balanceo local por documento (Optimizado con Pandas)
    muestras_none_balanceadas = []
    
    # Agrupamos los positivos por documento y categoría para obtener cuántos spans tiene cada una
    conteos_por_doc_cat = df_taxonomy.groupby(["id_documento", "categoria"]).size().reset_index(name="conteo")
    
    # Obtenemos el volumen MÁXIMO de spans entre todas las categorías para cada documento
    maximos_por_doc = conteos_por_doc_cat.groupby("id_documento")["conteo"].max().to_dict()

    # Iterar sobre los documentos que existen en el dataset positivo
    contratos_unicos = df_taxonomy["id_documento"].unique()

    for doc_id in contratos_unicos:
        # Obtener el límite máximo calculado para este documento específico
        volumen_maximo = maximos_por_doc.get(doc_id, 0)

        # Extraer los candidatos NONE disponibles para este documento
        candidatos_none = df_master_none[df_master_none["id_documento"] == doc_id]

        if not candidatos_none.empty and volumen_maximo > 0:
            # Seleccionar la cantidad exacta exigida por la regla (o el máximo disponible)
            n_samples = min(len(candidatos_none), volumen_maximo)
            
            # Muestreo aleatorio determinista utilizando la semilla global
            muestra = candidatos_none.sample(n=n_samples, random_state=RANDOM_SEED)
            muestras_none_balanceadas.append(muestra)

    # 4. Consolidación del dataset final
    if muestras_none_balanceadas:
        # Unir todos los fragmentos NONE seleccionados
        df_none_final = pd.concat(muestras_none_balanceadas, ignore_index=True)
        
        # Concatenar con los datos positivos originales
        df_consolidado = pd.concat([df_taxonomy, df_none_final], ignore_index=True)

        # 5. Guardar en disco
        # Sobrescribimos el archivo de taxonomía para que el flujo hacia split_dataset.py sea transparente
        df_consolidado.to_csv(DATASET_TAXONOMY_CSV, index=False, encoding="utf-8")
        
        # Reporte final por consola
        print("\n" + "="*50)
        print("     REPORTE DE INYECCIÓN (BALANCEO LOCAL)")
        print("="*50)
        print(f"Total registros positivos (originales): {len(df_taxonomy)}")
        print(f"Total registros NONE inyectados:      {len(df_none_final)}")
        print(f"Total dataset consolidado:            {len(df_consolidado)}")
        print("\nDistribución Final de Categorías:")
        print(df_consolidado["categoria"].value_counts())
        print("\n[ÉXITO] Inyección finalizada y guardada en dataset_taxonomy.csv.")
    else:
        print("\n[WARN] No se inyectaron muestras NONE. Revisa el archivo maestro.")


if __name__ == "__main__":
    run()