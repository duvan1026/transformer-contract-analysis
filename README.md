## Estructura del repositorio

```
project/
│
├── data/
│   ├── raw/
│   │   └── CUAD_v1.json
│   │   └── category_mapping.xlsx
│   │
│   └── processed/
│       └── dataset_spans.csv
│
├── docs/
│   ├── dataset_report.txt
│
├── src/
│   ├── config/
│   │   └── paths.py
│   │
│   ├── utils/
│   │   └── text_utils.py
│   │   └── category_utils.py
│   │
│   └── preprocessing/
│       └── extract_spans.py
│       └── validate_dataset.py
│       └── validate_dataset.py
│
└── README.md

```


### Descripción de carpetas

- **data/raw/**: contiene el dataset original sin modificar.
- **data/processed/**: contiene datasets generados tras el preprocesamiento.
- **docs/:** contiene reportes y documentación del dataset.
- **src/config/**: configuración global del proyecto (rutas y constantes)
- **src/utils/**:  funciones reutilizables de limpieza y manipulación de texto
- **src/preprocessing/**: scripts de procesamiento de datos.

---

## Extracción de Spans

El script `extract_spans.py` procesa el archivo JSON de CUAD siguiendo estos pasos:

1. Carga el JSON original del dataset.
2. Recorre cada contrato y sus párrafos.
3. Extrae las anotaciones de cada categoría.
4. Genera una fila por *span* anotado.
5. Maneja casos donde no existe respuesta (`is_impossible = True`).
6. Limpia el texto para evitar problemas de formato en CSV.
7. Guarda el dataset procesado en formato CSV.

### Ejecución:

```bash
python src/preprocessing/extract_spans.py
```

### Salida:

```
data/processed/dataset_spans.csv
```

## Formato del Dataset Generado

El archivo `dataset_spans.csv` usa un formato **long**, donde cada fila representa un span individual.

Columnas:

- **id_documento**: nombre del contrato
- **categoria**: categoría de la cláusula
- **question**: descripción de la categoría
- **is_impossible**: indica si la cláusula está ausente
- **span_text**: texto exacto del span
- **offset**: posición inicial del span en el documento

---

## Aplicación de la Taxonomía Contractual

El script apply_category_mapping.py aplica la taxonomía definida y genera el dataset final para entrenamiento.

Este proceso:

* filtra únicamente las categorías seleccionadas
* elimina registros sin spans válidos (is_impossible = True)
* unifica categorías según la taxonomía
* conserva la categoría original para trazabilidad
* elimina columnas no necesarias para el entrenamiento (question, is_impossible)

Ejecución:

```
python src/preprocessing/apply_category_mapping.py
```

Salida:

```
data/processed/dataset_taxonomy.csv
```
---

## Formato del Dataset Final (dataset_taxonomy.csv)

Este es el dataset utilizado para entrenar el modelo.

Cada fila representa un span válido.

Columnas:

* **id_documento** → identificador del contrato
* **categoria** → categoría final según la taxonomía
* **span_text** → texto exacto de la cláusula
* **offset** → posición dentro del documento
* **categoria_original** → categoría original en CUAD

Características:

* solo contiene spans válidos
* categorías completamente homologadas
* mantiene trazabilidad hacia la categoría original

---

## Validación del Dataset

El script validate_dataset.py genera un reporte con estadísticas del dataset final.

Incluye:

* número total de spans
* número total de contratos
* distribución de spans por categoría
* número de categorías presentes por contrato

Ejecución:

```
python src/evaluation/validate_dataset.py
```
Salida:

```
docs/dataset_report.txt
```

Este reporte permite verificar la calidad y cobertura del dataset antes del entrenamiento.

---

## Taxonomía Contractual

La taxonomía define las 5 categorías utilizadas en el entrenamiento del modelo:

* Limitation of Liability
* Audit Rights
* Governing Law
* Non-Compete
* Termination for Convenience

Incluye criterios semánticos y reglas de homologación desde las categorías originales de CUAD.

Para más detalles: **[Guía Detallada de Taxonomía Contractual](docs/taxonomy.md)**

---

## Mapeo de Etiquetas (Label Mapping)

El modelo de clasificación requiere que las categorías textuales se representen mediante identificadores numéricos. 

Para garantizar consistencia y reproducibilidad, el proyecto define un mapeo fijo entre cada categoría de la taxonomía y su identificador numérico.

Este mapeo se encuentra en:

```
src/config/label_mapping.py
```

Ejemplo:

```python
LABEL2ID = {
    "audit rights": 0,
    "governing law": 1,
    "limitation of liability": 2,
    "non-compete": 3,
    "termination for convenience": 4
}

ID2LABEL = {v: k for k, v in LABEL2ID.items()}
```

Este archivo:

* define el espacio oficial de clases del modelo
* asegura coherencia entre entrenamiento y evaluación
* permite interpretar correctamente las predicciones del modelo

**Importante**:
Si la taxonomía cambia, el mapeo debe actualizarse y considerarse una nueva versión experimental.

---

## Flujo completo de generación del dataset

Orden recomendado de ejecución:

```
# 1. Extraer spans desde CUAD
python src/preprocessing/extract_spans.py

# 2. Aplicar taxonomía y generar dataset final
python src/preprocessing/apply_category_mapping.py

# 3. Validar dataset generado
python src/evaluation/validate_dataset.py
```

Resultado final:

```
data/processed/dataset_taxonomy.csv
```

Este archivo está listo para la partición experimental y el entrenamiento del modelo BERT.

---
