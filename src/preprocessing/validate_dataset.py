from pathlib import Path
import sys
import pandas as pd

# Añade la raíz del proyecto al path de Python para poder importar módulos desde "src"
sys.path.append(str(Path(__file__).resolve().parents[2]))

# Rutas del dataset final y del archivo donde se guardará el reporte
from src.config.paths import (
    DATASET_TAXONOMY_CSV,
    DATASET_REPORT_FILE
)

spans = pd.read_csv(DATASET_TAXONOMY_CSV)

# Número total de fragmentos (spans)
total_spans = len(spans)

# Número de contratos distintos
total_docs = spans["id_documento"].nunique()

# Número de contratos distintos
spans_per_cat = spans["categoria"].value_counts()

# Cuántas categorías diferentes tiene cada contrato
cats_per_doc = spans.groupby("id_documento")["categoria"].nunique()

# Cuántas categorías diferentes tiene cada contrato
dist = cats_per_doc.value_counts().sort_index()

report = []

report.append("DATASET TAXONOMY REPORT")
report.append("="*40)
report.append(f"Total spans: {total_spans}")
report.append(f"Total contratos únicos: {total_docs}")
report.append("")

report.append("Spans por categoría:")
for cat, count in spans_per_cat.items():
    report.append(f"{cat}: {count}")

report.append("")
report.append("Contratos por número de categorías:")
for n, count in dist.items():
    report.append(f"{n} categorías: {count} contratos")

# Guarda el reporte en la carpeta docs
Path(DATASET_REPORT_FILE).write_text(
    "\n".join(report),
    encoding="utf-8"
)