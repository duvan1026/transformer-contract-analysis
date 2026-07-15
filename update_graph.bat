@echo off
echo ====================================================
echo [1/2] Re-indexando el repositorio (Actualizando Grafo)...
echo ====================================================
codebase-memory-mcp cli index_repository "{\"repo_path\": \"D:/DUVAN/Maestria/TFM/transformer-contract-analysis\"}"

echo.
echo ====================================================
echo [2/2] Exportando mapa completo de arquitectura en TXT...
echo ====================================================
:: Crear la carpeta de reportes si no existe
if not exist docs\architecture mkdir docs\architecture

:: Exportar el grafo al archivo TXT
codebase-memory-mcp cli search_graph "{\"project\": \"D-DUVAN-Maestria-TFM-transformer-contract-analysis\", \"query\": \"*\"}" > docs\architecture\complete_map.txt

echo.
echo ====================================================
echo ¡Proceso completado con exito!
echo Archivo actualizado en: docs/architecture/complete_map.txt
echo ====================================================
pause