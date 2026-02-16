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

### Ejecución

Para generar el dataset procesado:

```
python src/preprocessing/extract_spans.py
```

El script:
- lee el JSON desde **data/raw/**
- genera el CSV en **data/processed/**


