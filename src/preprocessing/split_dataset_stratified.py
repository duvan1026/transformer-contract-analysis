"""
Módulo para la partición del dataset utilizando Muestreo Aleatorio Estratificado Desproporcionado.
Garantiza una cuota mínima de documentos por categoría para proteger las clases minoritarias,
rellenando el resto de la muestra de forma aleatoria para favorecer la generalización.
"""

from pathlib import Path
import sys
import pandas as pd
import logging

# Añadir raíz del proyecto al path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config.paths import (
    DATASET_TAXONOMY_CSV,
    TRAIN_DATASET,
    VAL_DATASET,
    TEST_DATASET,
    SPLITS_DIR,
    SPLIT_REPORT_FILE
)

from src.config.training_config import (
    NUM_CONTRACTS,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
    MIN_SAMPLES_PER_CLASS
)

# Configuración básica de logging para feedback en consola
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def generate_split_report(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, contracts_selected: list) -> None:
    """
    Genera un reporte detallado con las estadísticas de la partición estratificada.
    """
    total_spans = len(train_df) + len(val_df) + len(test_df)
    report = []

    report.append("DATASET STRATIFIED SPLIT REPORT")
    report.append("=" * 40)
    report.append(f"Estrategia: Muestreo Estratificado Desproporcionado (Min {MIN_SAMPLES_PER_CLASS} docs/clase)")
    report.append(f"Total contratos seleccionados: {len(contracts_selected)}")
    report.append(f"Total spans: {total_spans}")
    report.append("")

    report.append("Spans por split")
    report.append("-" * 20)
    report.append(f"Train: {len(train_df)}")
    report.append(f"Validation: {len(val_df)}")
    report.append(f"Test: {len(test_df)}")
    report.append("")

    for split_name, df in zip(["Train", "Validation", "Test"], [train_df, val_df, test_df]):
        report.append(f"Distribución de categorías ({split_name})")
        report.append("-" * 35)
        for cat, count in df["categoria"].value_counts().items():
            report.append(f"{cat}: {count}")
        report.append("")

    SPLIT_REPORT_FILE.write_text("\n".join(report), encoding="utf-8")


def run() -> None:
    """
    Ejecuta el pipeline de partición garantizando la cobertura de clases minoritarias.
    """
    logging.info("Iniciando partición con muestreo estratificado desproporcionado...")

    # 1. Cargar dataset
    spans = pd.read_csv(DATASET_TAXONOMY_CSV)
    categorias_unicas = spans["categoria"].unique()
    
    contratos_seguros = set()

    # 2. FASE DE COBERTURA: Extraer cuota mínima por cada categoría
    for categoria in categorias_unicas:
        # Documentos que contienen esta categoría
        docs_con_categoria = spans[spans["categoria"] == categoria]["id_documento"].unique()
        
        # Determinar cuántos extraer (protección contra clases extremadamente pequeñas)
        n_extraer = min(MIN_SAMPLES_PER_CLASS, len(docs_con_categoria))
        
        # Muestreo aleatorio para esta cuota específica
        seleccionados = pd.Series(docs_con_categoria).sample(
            n=n_extraer, 
            random_state=RANDOM_SEED
        ).tolist()
        
        # Agregar al conjunto seguro (set automáticamente elimina duplicados)
        contratos_seguros.update(seleccionados)

    logging.info(f"Fase de Cobertura: {len(contratos_seguros)} contratos únicos asegurados para cubrir todas las clases.")

    # 3. FASE DE RELLENO ALEATORIO: Completar hasta NUM_CONTRACTS
    todos_los_contratos = set(spans["id_documento"].unique())
    contratos_restantes = list(todos_los_contratos - contratos_seguros)
    
    contratos_faltantes = NUM_CONTRACTS - len(contratos_seguros)
    
    if contratos_faltantes > 0:
        n_relleno = min(contratos_faltantes, len(contratos_restantes))
        contratos_relleno = pd.Series(contratos_restantes).sample(
            n=n_relleno, 
            random_state=RANDOM_SEED
        ).tolist()
    else:
        # Caso borde (Poco probable): Si la cuota mínima supera a NUM_CONTRACTS
        logging.warning("La cuota de seguridad superó el límite de NUM_CONTRACTS.")
        contratos_relleno = []
        contratos_seguros = set(list(contratos_seguros)[:NUM_CONTRACTS])

    # Unir ambas listas de selección
    contratos_seleccionados = list(contratos_seguros) + contratos_relleno
    logging.info(f"Fase de Relleno: {len(contratos_relleno)} contratos aleatorios añadidos. Total: {len(contratos_seleccionados)}.")

    # 4. FASE DE MEZCLA (SHUFFLE) Y PARTICIÓN (SPLIT)
    # Mezclar el arreglo final para destruir el orden de inyección
    contratos_finales = pd.Series(contratos_seleccionados).sample(
        frac=1, 
        random_state=RANDOM_SEED
    ).values

    # Calcular índices de corte
    train_size = int(len(contratos_finales) * TRAIN_RATIO)
    val_size = int(len(contratos_finales) * VAL_RATIO)

    train_contracts = contratos_finales[:train_size]
    val_contracts = contratos_finales[train_size : train_size + val_size]
    test_contracts = contratos_finales[train_size + val_size :]

    # Construir dataframes finales filtrando el dataset original
    train_df = spans[spans["id_documento"].isin(train_contracts)]
    val_df = spans[spans["id_documento"].isin(val_contracts)]
    test_df = spans[spans["id_documento"].isin(test_contracts)]

    # 5. GUARDAR RESULTADOS
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(TRAIN_DATASET, index=False)
    val_df.to_csv(VAL_DATASET, index=False)
    test_df.to_csv(TEST_DATASET, index=False)

    # 6. GENERAR REPORTE
    generate_split_report(train_df, val_df, test_df, contratos_seleccionados)

    logging.info("Partición generada y guardada correctamente:")
    logging.info(f" - Train: {len(train_df)} spans")
    logging.info(f" - Validation: {len(val_df)} spans")
    logging.info(f" - Test: {len(test_df)} spans")


if __name__ == "__main__":
    run()