## Estructura del repositorio

```
project-root/
│
├── data/
│   ├── raw/
│   │   └── CUAD_v1.json
│   └── processed/
│       └── dataset_spans.csv
│
├── src/
│   ├── config/
│   │   └── paths.py
│   │
│   ├── utils/
│   │   └── text_utils.py
│   │
│   └── preprocessing/
│       └── extract_spans.py
│
└── README.md

```


### Descripción de carpetas

- **data/raw/**: contiene el dataset original sin modificar.
- **data/processed/**: contiene datasets generados tras el preprocesamiento.
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

---

## Formato del Dataset Generado

El archivo `dataset_spans.csv` usa un formato **long**, donde cada fila representa un span individual.

Columnas:

- **id_documento**: nombre del contrato
- **categoria**: categoría de la cláusula
- **question**: descripción de la categoría
- **is_impossible**: indica si la cláusula está ausente
- **span_text**: texto exacto del span
- **offset**: posición inicial del span en el documento

Características del CSV:

- delimitador por comas
- comillas automáticas en todos los campos
- codificación UTF-8

Esto garantiza compatibilidad con Excel y herramientas de análisis.

---

## Taxonomía Contractual

El proceso de extracción se basa en una taxonomía técnica diseñada para estandarizar la identificación de cláusulas. Esta taxonomía define los criterios semánticos y patrones lingüísticos para 5 categorías clave, incluyendo la nueva estructura de **Limitación de Responsabilidad**.

Para consultar las definiciones detalladas, criterios de inclusión/exclusión y ejemplos reales de cada categoría, diríjase a:

**[Guía Detallada de Taxonomía Contractual](docs/taxonomy.md)**

---

## Guía de ejecución

Sigue los siguientes pasos para generar el dataset de trabajo a partir del dataset original CUAD:

### 1. Preparar el entorno

Asegúrate de tener instaladas las dependencias necesarias:

```bash
pip install pandas openpyxl
```

El archivo original debe estar ubicado en:

```
data/raw/CUAD_v1.json
```

Y el archivo de homologación de categorías en:

```
data/raw/category_mapping.xlsx
```

### 2. Extraer los spans desde CUAD

Este paso transforma el JSON original en un CSV donde cada fila corresponde a un fragmento (span) anotado.

```
python src/preprocessing/extract_spans.py
```

Salida generada:

```
data/processed/dataset_spans.csv
```

Este archivo contiene:

* id_documento
* categoria original de CUAD
* question
* is_impossible
* span_text
* offset

### 3. Aplicar la taxonomía y homologación de categorías

Este paso:

* filtra únicamente las categorías seleccionadas
* unifica categorías (por ejemplo, Cap on Liability y Uncapped Liability → Limitation of Liability)
* normaliza los nombres de categorías
* genera el dataset final para entrenamiento

```
python src/preprocessing/apply_category_mapping.py
```

Salida generada:

```
data/processed/dataset_taxonomy.csv
```

El archivo final contiene:

* id_documento
* categoria (según la taxonomía definida)
* categoria_original
* span_text
* offset
* question
* is_impossible

Cada fila representa un fragmento contractual etiquetado.

## ** To Do:** Documentar el archivo de configuracion para realizar el mapping de las categorias