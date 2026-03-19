# 1. Imagen base oficial de Python 3.11.5 (versión ligera)
# Debian Linux (llamada "slim").
FROM python:3.11.5-slim

# 2. Evitar que Python genere archivos .pyc y forzar logs en tiempo real
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Instalar dependencias esenciales del sistema operativo
# 'build-essential' es necesario para compilar algunas librerías de ML si fuera necesario
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Definir el directorio de trabajo dentro del contenedor
WORKDIR /app

# 5. Copiar el archivo de requisitos para instalar las librerías
COPY requirements.txt .

# 6. Instalar las dependencias de Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 7. Copiar el resto del código del proyecto al contenedor
COPY . .

# 8. Arreglar saltos de línea de Windows a Linux y dar permisos de ejecución
RUN sed -i 's/\r$//' run_pipeline.sh && chmod +x run_pipeline.sh

# 9. Comando por defecto para ejecutar tu pipeline completo
CMD ["./run_pipeline.sh"]