from pathlib import Path
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

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
    RANDOM_SEED
)

def generate_split_report(train_df, val_df, test_df, contracts_selected):
    """
    Genera un reporte con estadísticas del split del dataset.
    """

    total_spans = len(train_df) + len(val_df) + len(test_df)

    report = []

    report.append("DATASET SPLIT REPORT")
    report.append("=" * 40)
    report.append("")

    report.append(f"Total contratos seleccionados: {len(contracts_selected)}")
    report.append(f"Total spans: {total_spans}")
    report.append("")

    report.append("Spans por split")
    report.append("-" * 20)
    report.append(f"Train: {len(train_df)}")
    report.append(f"Validation: {len(val_df)}")
    report.append(f"Test: {len(test_df)}")
    report.append("")

    report.append("Distribución de categorías (Train)")
    report.append("-" * 35)
    for cat, count in train_df["categoria"].value_counts().items():
        report.append(f"{cat}: {count}")

    report.append("")
    report.append("Distribución de categorías (Validation)")
    report.append("-" * 35)
    for cat, count in val_df["categoria"].value_counts().items():
        report.append(f"{cat}: {count}")

    report.append("")
    report.append("Distribución de categorías (Test)")
    report.append("-" * 35)
    for cat, count in test_df["categoria"].value_counts().items():
        report.append(f"{cat}: {count}")

    SPLIT_REPORT_FILE.write_text(
        "\n".join(report),
        encoding="utf-8"
    )


def run():
    """
    Construye los datasets de entrenamiento, validación y prueba
    a partir del dataset de spans.

    El split se realiza por contrato (id_documento) para evitar
    que fragmentos del mismo contrato aparezcan en distintos
    conjuntos del experimento.
    """

    # Cargar dataset
    spans = pd.read_csv(DATASET_TAXONOMY_CSV)

    # Calcular número de categorías y spans por contrato
    stats = spans.groupby("id_documento").agg(
        n_categorias=("categoria", "nunique"),
        n_spans=("categoria", "count")
    ).reset_index()

    # Ordenar contratos por información disponible
    stats = stats.sort_values(
        ["n_categorias", "n_spans"],
        ascending=False
    )

    # Seleccionar los contratos más informativos
    selected_contracts = stats.head(NUM_CONTRACTS)["id_documento"]

    # Filtrar dataset con esos contratos
    dataset = spans[spans["id_documento"].isin(selected_contracts)]

    # Obtener lista de contratos
    contracts = dataset["id_documento"].unique()

    # Mezclar contratos para evitar sesgos
    contracts = pd.Series(contracts).sample(
        frac=1,
        random_state=RANDOM_SEED
    ).values

    # Calcular tamaños
    train_size = int(len(contracts) * TRAIN_RATIO)
    val_size = int(len(contracts) * VAL_RATIO)

    train_contracts = contracts[:train_size]
    val_contracts = contracts[train_size:train_size + val_size]
    test_contracts = contracts[train_size + val_size:]

    # Construir datasets
    train_df = dataset[dataset["id_documento"].isin(train_contracts)]
    val_df = dataset[dataset["id_documento"].isin(val_contracts)]
    test_df = dataset[dataset["id_documento"].isin(test_contracts)]

    # Crear carpeta si no existe
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    # Guardar datasets
    train_df.to_csv(TRAIN_DATASET, index=False)
    val_df.to_csv(VAL_DATASET, index=False)
    test_df.to_csv(TEST_DATASET, index=False)

    generate_split_report(
        train_df,
        val_df,
        test_df,
        selected_contracts
    )

    print("Split generado correctamente:")
    print("Train:", len(train_df))
    print("Validation:", len(val_df))
    print("Test:", len(test_df))


if __name__ == "__main__":
    run()