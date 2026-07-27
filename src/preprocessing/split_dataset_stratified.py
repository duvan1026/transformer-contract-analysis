"""
Módulo para la partición del dataset utilizando Muestreo Aleatorio Estratificado Desproporcionado.
Garantiza una cuota mínima de documentos por categoría para proteger las clases minoritarias,
rellenando el resto de la muestra de forma aleatoria para favorecer la generalización.
 
La partición final entre train/validation/test se realiza mediante estratificación
multi-etiqueta (MultilabelStratifiedShuffleSplit), de modo que la proporción de cada
categoría se mantenga equilibrada en los tres splits, no solo en el pool total.
"""

from pathlib import Path
import sys
import pandas as pd
import logging

from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

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


def validate_ratios() -> None:
    """
    Valida que las proporciones configurables de train/val/test sumen 1.0.
    Evita que un error de configuración pase silenciosamente y genere
    splits con contratos faltantes o duplicados.
    """
    suma_ratios = TRAIN_RATIO + VAL_RATIO + TEST_RATIO
    if not abs(suma_ratios - 1.0) < 1e-6:
        raise ValueError(
            f"Los ratios deben sumar 1.0, actualmente suman {suma_ratios} "
            f"(TRAIN={TRAIN_RATIO}, VAL={VAL_RATIO}, TEST={TEST_RATIO})"
        )


def contratos_por_categoria(df: pd.DataFrame) -> pd.Series:
    """
    Cuenta contratos únicos (no spans) que contienen cada categoría dentro de un split.
    A diferencia de value_counts() sobre spans, esto refleja cuántos documentos
    distintos aportan representación real de cada clase.
    """
    return df.groupby("categoria")["id_documento"].nunique().sort_values(ascending=False)


def generate_split_report(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, contracts_selected: list) -> None:
    """
    Genera un reporte detallado con las estadísticas de la partición estratificada,
    incluyendo tanto el conteo de spans como el conteo de contratos únicos por categoría.
    """
    total_spans = len(train_df) + len(val_df) + len(test_df)
    report = []
 
    report.append("DATASET STRATIFIED SPLIT REPORT")
    report.append("=" * 40)
    report.append(f"Estrategia: Muestreo Estratificado Desproporcionado (Min {MIN_SAMPLES_PER_CLASS} docs/clase)")
    report.append(f"Partición train/val/test: Multilabel Stratified Shuffle Split (iterstrat)")
    report.append(f"Ratios configurados: TRAIN={TRAIN_RATIO} | VAL={VAL_RATIO} | TEST={TEST_RATIO}")
    report.append(f"Total contratos seleccionados: {len(contracts_selected)}")
    report.append(f"Total spans: {total_spans}")
    report.append("")
 
    report.append("Spans por split")
    report.append("-" * 20)
    report.append(f"Train: {len(train_df)}")
    report.append(f"Validation: {len(val_df)}")
    report.append(f"Test: {len(test_df)}")
    report.append("")
 
    report.append("Contratos únicos por split")
    report.append("-" * 20)
    report.append(f"Train: {train_df['id_documento'].nunique()}")
    report.append(f"Validation: {val_df['id_documento'].nunique()}")
    report.append(f"Test: {test_df['id_documento'].nunique()}")
    report.append("")
 
    for split_name, df in zip(["Train", "Validation", "Test"], [train_df, val_df, test_df]):
        report.append(f"Distribución de categorías - SPANS ({split_name})")
        report.append("-" * 35)
        for cat, count in df["categoria"].value_counts().items():
            report.append(f"{cat}: {count}")
        report.append("")
 
        report.append(f"Distribución de categorías - CONTRATOS ({split_name})")
        report.append("-" * 35)
        for cat, count in contratos_por_categoria(df).items():
            report.append(f"{cat}: {count}")
        report.append("")
 
    SPLIT_REPORT_FILE.write_text("\n".join(report), encoding="utf-8")


def run() -> None:
    """
    Ejecuta el pipeline de partición garantizando la cobertura de clases minoritarias
    y una división final train/val/test estratificada a nivel multi-etiqueta.
    """
    logging.info("Iniciando partición con muestreo estratificado desproporcionado...")

    # 0. Validar configuración de ratios antes de cualquier muestreo
    validate_ratios()

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
    contratos_restantes = sorted(list(todos_los_contratos - contratos_seguros))
    
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

    # 4. FASE DE PARTICIÓN ESTRATIFICADA MULTI-ETIQUETA (train / val / test)
    # Se construye una matriz binaria contrato x categoría (presencia/ausencia,
    # no conteo de spans) para que iterstrat balancee por presencia de clase.
    df_pool = spans[spans["id_documento"].isin(contratos_seleccionados)]
    matriz = pd.crosstab(df_pool["id_documento"], df_pool["categoria"])
    matriz = (matriz > 0).astype(int)

    ids = matriz.index.values
    Y = matriz.values

    # Proporción de (val + test) sobre el total, y proporción de test dentro de ese resto.
    # Se derivan de las variables configurables, no de valores fijos.
    resto_ratio = VAL_RATIO + TEST_RATIO
    test_size_dentro_de_resto = TEST_RATIO / resto_ratio

    # Corte 1: separa train del resto (val + test)
    msss_1 = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=resto_ratio, random_state=RANDOM_SEED
    )
    train_idx, resto_idx = next(msss_1.split(ids, Y))
 
    ids_resto = ids[resto_idx]
    Y_resto = Y[resto_idx]
 
    # Corte 2: separa val de test dentro del resto
    msss_2 = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=test_size_dentro_de_resto, random_state=RANDOM_SEED
    )
    val_idx, test_idx = next(msss_2.split(ids_resto, Y_resto))
 
    train_contracts = ids[train_idx]
    val_contracts = ids_resto[val_idx]
    test_contracts = ids_resto[test_idx]
 
    logging.info(
        f"Partición estratificada: Train={len(train_contracts)} | "
        f"Validation={len(val_contracts)} | Test={len(test_contracts)} contratos."
    )
 
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