# Clasificación de Cláusulas Contractuales con BERT

Este proyecto representa la implementación de un pipeline de procesamiento de lenguaje natural (NLP) **LegalTech** para la clasificación automática de cláusulas legales. Utilizando el dataset académico **CUAD (Contract Understanding Atticus Dataset)**, se ha diseñado un flujo de trabajo que abarca desde la extracción cruda de datos hasta el entrenamiento, la auditoría detallada y la inferencia en producción de un modelo basado en la arquitectura **BERT** (`bert-base-uncased`, 110M de parámetros).

El diseño del sistema responde a dos principios no negociables en un contexto legal-corporativo:

- **Zero-Cloud / Procesamiento 100% local.** Todo el pipeline —desde la fragmentación de texto hasta la inferencia del modelo— se ejecuta íntegramente en infraestructura local o en un contenedor Docker aislado, sin invocar ninguna API externa de terceros. Ningún fragmento de un contrato abandona el entorno controlado, un requisito imprescindible dado el **secreto profesional** y la confidencialidad propia de la documentación legal y corporativa (en línea con los principios de minimización y control de datos de normativas como GDPR).
- **Green AI.** En lugar de recurrir a modelos generativos de propósito general, el sistema realiza *fine-tuning* de una arquitectura deliberadamente ligera y auditable (BERT, 110M de parámetros), reduciendo la huella de cómputo frente a alternativas basadas en LLMs de mayor escala.

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
iterative-stratification==0.1.9
```

## 1. Estructura del repositorio

El repositorio se ha organizado siguiendo principios de modularidad para separar claramente la lógica de datos, la configuración y los resultados:

```
project/
│
├── data/
│   ├── raw/                 # Dataset original CUAD (CUAD_v1.json) y mapeo de taxonomía (category_mapping.xlsx)
│   ├── processed/           # Datasets tras limpieza, inyección de NONE y particiones (splits)
│   │   ├── splits/          # train.csv, validation.csv, test.csv
│   │   └── inference/       # Lotes de fragmentos y resultados de la fase de producción
│   └── metadata/            # Registros Maestros de trazabilidad documental (baseline / balanced)
│
├── docs/                    # Documentación y reportes de resultados
│   └── reports/             # Artefactos de evaluación (Matriz, Métricas, Auditoría)
│
├── models/                  # Pesos y checkpoints del modelo BERT entrenado
│   ├── bert_baseline/       # Artefactos del experimento Baseline
│   └── bert_balanced/       # Artefactos del experimento Balanced
│
├── src/
│   ├── config/               # Configuración centralizada (rutas, mapeo de etiquetas, estándares de texto)
│   ├── utils/                 # Fragmentación, filtros de calidad, utilidades espaciales y de categorías
│   ├── preprocessing/         # Scripts de preparación de datos (extracción, taxonomía, splits, clase NONE)
│   ├── training/               # Lógica de entrenamiento y definición del Dataset PyTorch
│   ├── evaluation/             # Scripts de validación e inferencia de test
│   ├── data/                   # Inicialización del Registro Maestro de documentos
│   └── inference/              # Pipeline de producción (Fragmentación y Scoring con IA)
│
├── Dockerfile
├── requirements.txt
├── run_pipeline.sh
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

La homologación **no está hardcodeada** en el código: se define en un archivo Excel externo (`data/raw/category_mapping.xlsx`), con las columnas `Category`, `Usar` y `Final Name Category`. Solo las filas marcadas con `Usar = "Sí"` se incorporan al mapeo final, lo que permite ajustar la taxonomía sin tocar el código fuente.

Incluye criterios semánticos y reglas de homologación desde las categorías originales de CUAD.Para más detalles: **[Guía Detallada de Taxonomía Contractual](docs/taxonomy.md)**

### 2.2 Centralización del Mapeo (label_mapping.py)

para garantizar la consistencia, se ha implementado una "fuente única de verdad" que gestiona:

* **MODEL_NAME**: `bert-base-uncased` (110M de parámetros).
* **Conversión de Etiquetas**: dos mapeos fijos independientes —`LABEL2ID_BASELINE` (5 clases) y `LABEL2ID_BALANCED` (6 clases, incluye `NONE`)— que aseguran que el modelo siempre asocie el mismo número a la misma categoría legal, evitando errores de desalineación entre el entrenamiento y la inferencia.
* **Escalabilidad**: el número de etiquetas (`NUM_LABELS_BASELINE` / `NUM_LABELS_BALANCED`) se calcula dinámicamente a partir de cada diccionario, facilitando la futura expansión de la taxonomía.

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

### C. Generación de la Clase `NONE` (exclusivo de la rama Balanced)

Dos pasos adicionales, ejecutados únicamente en el experimento `balanced`, construyen una clase negativa explícita antes del split:

* **`generate_none_class.py`**: recorre el texto completo de los contratos válidos y extrae candidatos `NONE` aplicando un embudo de filtros — longitud mínima de 40 caracteres (para evitar *shortcut learning*), densidad alfabética mínima del 70% (estándar inspirado en Gopher/DeepMind y ROOTS/BigScience), y exclusión espacial de cualquier zona que se solape con un span positivo ya anotado. Salida: `data/processed/dataset_none_master.csv`.
* **`inject_none_class.py`**: aplica un balanceo **local por documento** — para cada contrato, muestrea aleatoriamente tantos fragmentos `NONE` como el volumen máximo de spans de cualquier categoría positiva dentro de ese mismo contrato — y sobrescribe `dataset_taxonomy.csv` con el dataset consolidado (positivos + `NONE`).

### D. Validación del Dataset (validate_dataset.py)

Genera estadísticas de distribución para verificar la calidad antes del entrenamiento.

* **Salida**: docs/dataset_report.txt (Reporte de balances y conteos).

### E. Generación de Splits

La estrategia de partición difiere según la rama experimental:

* **Baseline — `split_dataset_density.py`**: selecciona los `NUM_CONTRACTS` contratos con mayor densidad de información (más categorías y más spans) y los divide por contrato completo en train/val/test (`70% / 15% / 15%`), evitando fuga de datos entre splits.
* **Balanced — `split_dataset_stratified.py`**: garantiza primero una cuota mínima de `15` documentos por categoría (incluida `NONE`) para proteger a las clases minoritarias, completa la selección hasta `NUM_CONTRACTS` con relleno aleatorio, y realiza la partición final train/val/test mediante **`MultilabelStratifiedShuffleSplit`** (librería `iterstrat`) sobre una matriz binaria contrato × categoría, manteniendo la proporción de cada clase en los tres splits.

* **Salidas (ambas ramas)**: `data/processed/splits/` (`train.csv`, `validation.csv`, `test.csv`, `split_report.txt`).

### Manejo de textos largos: Chunking Determinista

Dado que BERT opera con una ventana física máxima de **512 tokens** (`MAX_MODEL_TOKENS`), tanto la generación de la clase `NONE` como la fase de inferencia sobre documentos nuevos (Sección 6) utilizan un motor de **fragmentación determinista basada en párrafos**, no una ventana deslizante (*sliding window*):

1. Se segmenta el contrato en párrafos naturales, conservando los *offsets* absolutos de carácter.
2. Si el párrafo cabe en el límite de tokens, se conserva íntegro como un único fragmento.
3. Si lo excede, se subdivide acumulando oraciones completas hasta el umbral máximo, sin cortar frases a mitad de camino.
4. Cada fragmento candidato pasa por un filtro de calidad (longitud mínima, contenido no puramente numérico); los descartados quedan registrados en un CSV de auditoría independiente, sin pérdida silenciosa de información.

---

## 4. Entrenamiento y Modelado

En esta fase se realiza el ajuste fino (*fine-tuning*) del modelo base BERT para la tarea específica de clasificación de texto legal. Debido al tamaño de los archivos binarios, los pesos del modelo no se incluyen en el repositorio; el sistema está diseñado para generar estos artefactos localmente tras la ejecución del script de entrenamiento.

* **Ejecución**:
  * Baseline: `python src/training/train_bert_baseline.py`
  * Balanced: `python src/training/train_bert_balanced.py`
* **Salida**: los artefactos se generan en `models/bert_baseline/` o `models/bert_balanced/` según corresponda, listos para ser consumidos por el script de evaluación.

Ambos scripts comparten los mismos hiperparámetros (`learning_rate=2e-5`, `batch_size=8`, `3 épocas`, `weight_decay=0.01`) y usan **F1-Macro** como criterio de selección del mejor checkpoint (`metric_for_best_model="f1"`), priorizado sobre el Accuracy por su robustez frente al desbalance de clases típico de corpus jurídicos.

**Diferencia central — función de pérdida:**

| | Baseline | Balanced |
|---|---|---|
| Clases | 5 | 6 (incluye `NONE`) |
| `Trainer` | Estándar de Hugging Face | `BalancedTrainer` personalizado (subclase de `Trainer`) |
| Función de pérdida | `nn.CrossEntropyLoss()` sin ponderar | `nn.CrossEntropyLoss(weight=class_weights)`, con pesos calculados vía `sklearn.utils.class_weight.compute_class_weight(class_weight="balanced")` sobre las etiquetas reales de entrenamiento |

---

## 5. Evaluación y Auditoría de Resultados

Se ha desarrollado un pipeline (`src/evaluation/evaluate_test.py --model {baseline|balanced}`) que genera informes detallados en `docs/reports/`, con nombres de archivo diferenciados por experimento:

* **Reporte de Métricas** (`test_metrics_report_baseline.txt` / `test_metrics_report_balanced.txt`): contiene Precision, Recall y F1-Score por clase (`classification_report` de scikit-learn), junto con un glosario explicativo de cada métrica. Se prioriza el **F1-Macro** como indicador global de éxito.
* **Matriz de Confusión** (`confusion_matrix_baseline.png` / `confusion_matrix_balanced.png`): visualización (`seaborn.heatmap`) con ejes etiquetados en inglés (*True Label* vs *Predicted Label*) para identificar ambigüedades entre cláusulas.
* **Auditoría Detallada** (`test_detailed_baseline.csv` / `test_detailed_balanced.csv`): informe fila por fila que permite una revisión cualitativa.
  * **Columnas**: incluye el texto de la cláusula, categoría real, categoría predicha, y `confidence_pct` (confianza formateada para lectura en Excel).

---

## 6. Fase de Inferencia en Producción (Framework)

Este proyecto no solo entrena un modelo, sino que funciona como un **framework** listo para clasificar nuevos contratos masivamente. Se ha implementado un sistema de trazabilidad basado en un **Registro Maestro** (`master_registry_baseline.csv` / `master_registry_balanced.csv`, uno por experimento) que controla el ciclo de vida de cada documento (`Disponible` → `En_proceso` → `Utilizado` → `Clasificado`).

* **Paso 0: Inicialización del Inventario** (`src/data/00_init_master_registry.py --model {baseline|balanced}`)
  Rastrea todos los contratos disponibles en CUAD y los splits de entrenamiento previos para establecer el estado inicial del sistema, aislado por experimento.

* **Paso 1: Preparación de Lotes** (`src/inference/01_prepare_batch_documents.py --model {baseline|balanced}`)
  Toma un lote de contratos en estado `Disponible` (tamaño configurable, `BATCH_SIZE_DOCS`), los fragmenta mediante el motor de *chunking* determinista (respetando el límite de 512 tokens de BERT) y genera el archivo de carga (`batch_fragments.csv`), reservando los documentos y marcándolos como `Utilizado` al finalizar.

* **Paso 2: Inferencia Pura** (`src/inference/02_run_inference.py --model {baseline|balanced}`)
  Ejecuta el modelo BERT correspondiente sobre los fragmentos preparados, calculando la matriz de probabilidades vía Softmax y la predicción cruda (`argmax`). Actúa exclusivamente como motor matemático: **no aplica ninguna regla de negocio ni umbral** en este paso — esa responsabilidad se delega a la fase siguiente para permitir simulaciones eficientes sin reprocesamiento neuronal. Salida: `inference_scored_results.csv`; los documentos quedan marcados como `Clasificado`.

* **Paso 3: Cálculo de Ground Truth** (`src/inference/03_calculate_ground_truth.py --model {baseline|balanced}`)
  Cruza las predicciones con las anotaciones originales de CUAD mediante superposición espacial de *offsets* (umbral mínimo de 30 caracteres solapados), generando la columna `etiqueta_real` de forma inmutable. Salida: `ground_truth_results.csv`.

* **Paso 4: Reglas de Negocio y Estructuración JSON** (`src/inference/04_generate_reports.py --model {baseline|balanced}`)
  Aplica la lógica asimétrica por experimento:
  * **Baseline**: evalúa el rendimiento frente a múltiples umbrales de confianza (`0.70` a `0.95`) para simular una regla de descarte (`NONE` implícito) que mitigue los Falsos Positivos, generando un set completo de reportes **por cada umbral**.
  * **Balanced**: al existir ya una clase `NONE` entrenada explícitamente en el modelo, no aplica umbral externo — usa la predicción cruda directamente (confianza plena) y añade una distribución de confianza al reporte.

  Por cada configuración procesada, genera: un reporte de métricas detallado (F1-Macro, Precision, Recall a nivel fragmento), una matriz de confusión en formato imagen, y una **salida estructurada en JSON** (`json_outputs/`) por cada contrato clasificado.


### 6.1 Formato de Salida Estructurada (JSON)
El pipeline convierte los párrafos clasificados en representaciones estructuradas por contrato. Ejemplo de salida generada en la carpeta `json_outputs/`:

```json
{
    "document_id": "ABILITYINC_SERVICES_AGREEMENT",
    "processing_date": "2026-05-21T21:32:40Z",
    "applied_threshold": 0.9,
    "extracted_clauses": {
        "audit rights": [],
        "governing law": [
            {
                "score": 0.9845,
                "text_snippet": "This Agreement is governed by the laws of the State of Delaware...",
                "start_offset": 24546,
                "end_offset": 26116
            }
        ],
        "limitation of liability": [],
        "non-compete": [],
        "termination for convenience": []
    }
}
```

---

## 7. Flujo de Ejecución Recomendado

```bash
# 1. Ingeniería de Datos
python src/preprocessing/extract_spans.py
python src/preprocessing/apply_category_mapping.py

# 1.b [Solo rama Balanced] Generación de la clase NONE
python src/preprocessing/generate_none_class.py
python src/preprocessing/inject_none_class.py

# 2. Análisis y Partición
python src/preprocessing/validate_dataset.py
python src/preprocessing/split_dataset_density.py       # Baseline
# o bien:
python src/preprocessing/split_dataset_stratified.py    # Balanced

# 3. Modelado (Entrenamiento)
python src/training/train_bert_baseline.py               # Baseline
# o bien:
python src/training/train_bert_balanced.py                # Balanced

# 4. Evaluación y Auditoría Final
python src/evaluation/evaluate_test.py --model baseline
# o bien:
python src/evaluation/evaluate_test.py --model balanced

# 5. Gestión de Inventario y Estado (Gateway a Producción)
python src/data/00_init_master_registry.py --model baseline

# 6. Pipeline de Inferencia (Clasificación en Producción)
python src/inference/01_prepare_batch_documents.py --model baseline
python src/inference/02_run_inference.py --model baseline
python src/inference/03_calculate_ground_truth.py --model baseline
python src/inference/04_generate_reports.py --model baseline
```

> ⚠️ Este flujo manual es equivalente, paso a paso, a lo que ejecuta automáticamente `run_pipeline.sh` (Sección 8). Para la mayoría de los casos de uso, se recomienda **el flujo Docker orquestado** en lugar de la ejecución manual.

## 8. Reproducibilidad Total con Docker (Método Recomendado)

Al entrenar el modelo en distintos equipos durante la investigación de este TFM, noté ligeras variaciones en el Accuracy general (alrededor del 1.33%). Al investigar, concluí que esto se debe a diferencias en la arquitectura física del hardware y la gestión de hilos del sistema operativo.

Para solucionar esto, salir de dudas y garantizar un ambiente 100% estándar e independiente de la máquina, decidí dockerizar la solución completa.

Al usar este método, el código corre bajo un sistema Linux aislado y controlado, pero guarda los archivos pesados (modelos) y los resultados (CSVs, imágenes) directamente en tu computador usando volúmenes. Este aislamiento refuerza además la política **Zero-Cloud**: el contenedor no requiere ni realiza ninguna llamada de red saliente para procesar los contratos.

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

### Paso 2: Ejecutar el pipeline

Lanza el contenedor mapeando las carpetas de datos, modelos, reportes y logs
con tu equipo local. El script orquestador `run_pipeline.sh` **requiere
obligatoriamente** uno de estos 5 valores: `baseline`, `balanced`,
`build_holdout`, `baseline_holdout`, `balanced_holdout`; si se omite o es
inválido, el pipeline se detiene con un mensaje de uso.

**En PowerShell (Windows):**

```powershell
# 1. Ciclo normal de entrenamiento y evaluación — ejecutar ambas lanes primero
docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/models:/app/models" -v "$(pwd)/docs/reports:/app/docs/reports" -v "$(pwd)/logs:/app/logs" pipeline-tfm ./run_pipeline.sh baseline

docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/models:/app/models" -v "$(pwd)/docs/reports:/app/docs/reports" -v "$(pwd)/logs:/app/logs" pipeline-tfm ./run_pipeline.sh balanced

# 2. Construcción del lote final de holdout (100 contratos, una sola vez,
#    SOLO después de que ambas lanes anteriores ya hayan corrido)
docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/models:/app/models" -v "$(pwd)/docs/reports:/app/docs/reports" -v "$(pwd)/logs:/app/logs" pipeline-tfm ./run_pipeline.sh build_holdout

# 3. Evaluación definitiva sobre el lote de holdout, en cualquier orden
docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/models:/app/models" -v "$(pwd)/docs/reports:/app/docs/reports" -v "$(pwd)/logs:/app/logs" pipeline-tfm ./run_pipeline.sh baseline_holdout

docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/models:/app/models" -v "$(pwd)/docs/reports:/app/docs/reports" -v "$(pwd)/logs:/app/logs" pipeline-tfm ./run_pipeline.sh balanced_holdout
```

**En Bash (Linux/Mac):** mismos comandos, sustituyendo `$(pwd)` según tu shell habitual.

- `baseline` / `balanced`: corren las 4 fases completas (limpieza →
  ingeniería de datos → entrenamiento/evaluación → inferencia normal).
- `build_holdout`: construye el lote final de 100 contratos comunes a
  ambas lanes. Se ejecuta **una sola vez**; si los archivos ya existen,
  el script se detiene con un mensaje explícito para evitar sobrescribir
  una selección ya consumida por una inferencia de holdout.
- `baseline_holdout` / `balanced_holdout`: ejecutan únicamente la
  inferencia (sin limpieza, sin reentrenamiento) sobre el lote fijo de
  holdout, usando el modelo ya entrenado de la lane correspondiente.


## 9. Próximos Pasos (Roadmap)
* **Fase 2 (En desarrollo):** Transición de un modelo de 5 clases a 6 clases, incorporando explícitamente la categoría `NONE`. Esto permitirá al modelo aprender semánticamente a descartar texto irrelevante o cláusulas estándar (ej. definiciones) sin depender exclusivamente de filtros por umbral de confianza, mitigando el problema del desbalance de clases y reduciendo drásticamente los Falsos Positivos de alta confianza.