# Clasificación de Cláusulas Contractuales con BERT

Este proyecto representa la implementación de un pipeline de procesamiento de lenguaje natural (NLP) para la clasificación automática de cláusulas legales. Utilizando el dataset CUAD, se ha diseñado un flujo de trabajo que abarca desde la extracción cruda de datos hasta el entrenamiento y la auditoría detallada de un modelo basado en la arquitectura BERT.

## 0. Requisitos y Configuración del Entorno

Para garantizar la reproducibilidad de los resultados y evitar conflictos de dependencias, se ha estandarizado el uso de Python 3.11.5.

### 0.1 Instalar Python 3.11.5

Si no tienes esta versión, descárgala e instálala desde el sitio oficial:

* [Descargar Python 3.11.5 para Windows/Mac/Linux](https://www.python.org/downloads/release/python-3115/)

Nota en Windows: Asegúrate de marcar la casilla "Add Python to PATH" durante la instalación.

### A. Preparación del Ambiente Virtual

Se recomienda encarecidamente el uso de un entorno virtual (venv) dentro de la carpeta del proyecto para aislar las librerías:

```
# 1. Crear el entorno virtual (asegúrate de tener Python 3.11 instalado en el equipo)
py -3.11 -m venv .venv

# 2. Activar el entorno
# En Windows:
.venv\Scripts\activate
# En Linux/Unix (Supercomputadora):
source .venv/bin/activate

# 3. Actualizar pip e instalar dependencias
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Nota de verificación:** Para confirmar que estás utilizando la versión correcta en tu entorno, ejecuta `python --version` en tu terminal.

### B. Archivo de Dependencias (requirements.txt)

El proyecto utiliza las siguientes versiones base para asegurar la compatibilidad con los nodos del clúster:

```
torch==2.1.0
transformers==4.35.0
tokenizers==0.14.1
accelerate==0.25.0
huggingface_hub==0.17.3
scikit-learn==1.3.0
pandas==2.1.1
numpy==1.26.0
matplotlib==3.8.0
seaborn==0.13.0
openpyxl==3.1.5
```

## 1. Estructura del repositorio

El repositorio se ha organizado siguiendo principios de modularidad para separar claramente la lógica de datos, la configuración y los resultados:

```
project/
│
├── data/
│   ├── raw/                # Dataset original CUAD y mapeos base
│   └── processed/          # Datasets tras limpieza y particiones (splits)
│
├── docs/                   # Documentación y reportes de resultados
│   └── reports/            # Artefactos de evaluación (Matriz, Métricas, Auditoría)
│
├── models/                 # Pesos y checkpoints del modelo BERT entrenado
│
├── src/
│   ├── config/             # Configuración centralizada (Rutas y Metadatos del modelo)
│   ├── preprocessing/      # Scripts de preparación de datos
│   ├── training/           # Lógica de entrenamiento y definición del Dataset PyTorch
│   └── evaluation/         # Scripts de validación e inferencia de test
│   └── inference/          # Pipeline de producción (Fragmentación y Scoring con IA)
│
└── README.md

```

## 2. Taxonomía Contractual y Mapeo de Etiquetas

Un pilar fundamental de este desarrollo es la definición de una Taxonomía Contractual optimizada. 

### 2.1 Definición de Categorías

Se han homologado las categorías originales de CUAD en 5 clases críticas para el análisis legal:

* Limitation of Liability
* Audit Rights
* Governing Law
* Non-Compete
* Termination for Convenience

Incluye criterios semánticos y reglas de homologación desde las categorías originales de CUAD.Para más detalles: **[Guía Detallada de Taxonomía Contractual](docs/taxonomy.md)**

### 2.2 Centralización del Mapeo (label_mapping.py)

para garantizar la consistencia, se ha implementado una "fuente única de verdad" que gestiona:

* **MODEL_NAME**: bert-base-uncased (110M de parámetros).
* **Conversión de Etiquetas**: Un mapeo fijo (LABEL2ID / ID2LABEL) que asegura que el modelo siempre asocie el mismo número a la misma categoría legal, evitando errores de desalineación entre el entrenamiento y la inferencia.
* **Escalabilidad**: El número de etiquetas (NUM_LABELS) se calcula dinámicamente, facilitando la futura expansión de la taxonomía.

**Importante**: Si la taxonomía cambia, el mapeo debe actualizarse y considerarse una nueva versión experimental.

---

## 3. Fase de Ingeniería de Datos

### A. Extracción de Spans (extract_spans.py)

Procesa el JSON original de CUAD para extraer fragmentos de texto anotados.

* **Salida**: data/processed/dataset_spans.csv

* **Columnas**: id_documento, categoria, question, is_impossible, span_text, offset.

### B. Aplicación de Taxonomía (apply_category_mapping.py)

Filtra las categorías seleccionadas y unifica la taxonomía final.

* **Salida**: data/processed/dataset_taxonomy.csv

* **Columnas**: id_documento, categoria (final), span_text, offset, categoria_original.

### C. Validación del Dataset (validate_dataset.py)

Genera estadísticas de distribución para verificar la calidad antes del entrenamiento.

* **Salida**: docs/dataset_report.txt (Reporte de balances y conteos).

### D. Generación de Splits (split_dataset.py)

Divide los datos en entrenamiento, validación y prueba de forma reproducible.

* **Salidas**: data/processed/splits/ (train.csv, validation.csv, test.csv).

---

## 4. Entrenamiento y Modelado

En esta fase se realiza el ajuste fino (Fine-tuning) del modelo base BERT para la tarea específica de clasificación de texto legal. Debido al tamaño de los archivos binarios, los pesos del modelo no se incluyen en el repositorio; el sistema está diseñado para generar estos artefactos localmente tras la ejecución del script de entrenamiento.

* **Ejecución**: python src/training/train_bert.py

* **Salida**: Los artefactos se generan en la carpeta models/. Esta carpeta contendrá el modelo entrenado listo para ser utilizado por el script de evaluación.

---

## 5. Evaluación y Auditoría de Resultados

Se ha desarrollado un pipeline que genera informes detallados en docs/reports/:

* **Reporte de Métricas (test_metrics_report.txt)**: Contiene Precision, Recall y F1-Score por clase. Se prioriza el F1-Macro como indicador global de éxito.

* **Matriz de Confusión (confusion_matrix.png)**: Visualización con ejes etiquetados en inglés (True Label vs Predicted Label) para identificar ambigüedades entre cláusulas.

* **Auditoría Detallada (test_detailed_results.csv)**:Informe fila por fila que permite una revisión cualitativa.
    * **Columnas**: Incluye el texto de la cláusula, categoría real, categoría predicha, y confidence_pct (Confianza formateada para lectura en Excel).

---

## 6. Fase de Inferencia en Producción (Framework)

Este proyecto no solo entrena un modelo, sino que funciona como un framework listo para clasificar nuevos contratos masivamente. Se ha implementado un sistema de trazabilidad basado en un Registro Maestro (`master_document_registry.csv`) que controla el ciclo de vida de cada documento (`Disponible` -> `Utilizado` -> `Clasificado`).

* **Paso 0: Inicialización del Inventario (`src/data/00_init_master_registry.py`)**
  Rastrea todos los contratos disponibles y los splits de entrenamiento previos para establecer el estado inicial del sistema.

* **Paso 1: Preparación de Lotes (`src/inference/01_prepare_batch_documents.py`)**
  Toma los contratos en estado `Disponible`, los divide en párrafos y fragmentos (respetando la longitud máxima de tokens de BERT) y genera un archivo de carga para la GPU (`batch_fragments.csv`), cambiando el estado de los documentos a `Utilizado`.

* **Paso 2: Clasificación y Reglas de Negocio (`src/inference/02_run_inference.py`)**
  Orquesta el modelo BERT para analizar masivamente los fragmentos. Aplica un umbral de confianza estricto (ej. 75%). Si la IA no supera este umbral, el fragmento se envía a una categoría `NONE` para revisión manual. Al finalizar, guarda el dataset enriquecido (`inference_scored_results.csv`) y marca los contratos como `Clasificado`.

---

## 7. Flujo de Ejecución Recomendado

```
# 1. Ingeniería de Datos
python src/preprocessing/extract_spans.py
python src/preprocessing/apply_category_mapping.py

# 2. Análisis y Partición
python src/preprocessing/validate_dataset.py
python src/preprocessing/split_dataset.py

# 3. Modelado (Entrenamiento)
python src/training/train_bert.py

# 4. Evaluación y Auditoría Final
python src/evaluation/evaluate_test.py

# 5. Gestión de Inventario y Estado (Gateway a Producción)
python src/data/00_init_master_registry.py

# 6. Pipeline de Inferencia (Clasificación en Producción)
python src/inference/01_prepare_batch_documents.py
python src/inference/02_run_inference.py
```

## 8. Reproducibilidad Total con Docker (Método Recomendado)

Al entrenar el modelo en distintos equipos durante la investigación de este TFM, noté ligeras variaciones en el Accuracy general (alrededor del 1.33%). Al investigar, concluí que esto se debe a diferencias en la arquitectura física del hardware y la gestión de hilos del sistema operativo.

Para solucionar esto, salir de dudas y garantizar un ambiente 100% estándar e independiente de la máquina, decidí dockerizar la solución completa.

Al usar este método, el código corre bajo un sistema Linux aislado y controlado, pero guarda los archivos pesados (modelos) y los resultados (CSVs, imágenes) directamente en tu computador usando volúmenes.

### Requisitos Previos para este método
Antes de comenzar, asegúrate de cumplir con lo siguiente en tu equipo:
1. **Instalar Docker Desktop:** Si no lo tienes, descárgalo e instálalo desde [su página oficial](https://www.docker.com/products/docker-desktop/).
2. **Motor en ejecución:** Asegúrate de abrir la aplicación Docker Desktop y esperar a que indique que el motor está corriendo (*Engine running*) antes de ejecutar cualquier comando en la terminal.
3. **WSL2 (Solo Windows):** Durante la instalación en Windows, asegúrate de marcar la opción para usar WSL2 en lugar de Hyper-V, ya que ofrece un rendimiento muy superior para Machine Learning.

### Paso 0: Asignar recursos a Docker (Solo para Windows/WSL2)

El entrenamiento del modelo BERT exige un alto consumo de memoria RAM y CPU. Si usas Windows, Docker limita estos recursos por defecto, lo que puede causar un error de falta de memoria (OOM). Para solucionarlo, debes indicarle a Windows que le dé más potencia a la máquina virtual:

1. Abre el Explorador de archivos y ve a tu carpeta de usuario (escribe `%USERPROFILE%` en la barra de direcciones).
2. Crea (o edita) un archivo llamado `.wslconfig`.
3. Pega el siguiente contenido (ajusta los valores según la capacidad total de tu equipo):

```
ini
[wsl2]
memory=12GB 
processors=6
swap=8GB
```

4. Abre PowerShell y ejecuta wsl --shutdown para aplicar los cambios. Luego vuelve a abrir Docker Desktop.

### Paso 1: Construir la imagen

Abre tu terminal en la raíz de este proyecto y empaqueta el entorno:

```
docker build -t pipeline-tfm .
```

### Paso 2: Ejecutar el pipeline completo

Lanza el contenedor mapeando las carpetas de datos, modelos y reportes con tu equipo local:

En PowerShell (Windows):

```
docker run --rm -v "${PWD}/data:/app/data" -v "${PWD}/models:/app/models" -v "${PWD}/docs/reports:/app/docs/reports" pipeline-tfm
```

En Bash (Linux/Mac):

```
docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/models:/app/models" -v "$(pwd)/docs/reports:/app/docs/reports" pipeline-tfm
```
