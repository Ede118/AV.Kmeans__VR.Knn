# Cambios realizados
- `Code/adapters/Repositorio.py`: normalicé la persistencia (k-means/kNN/json), corregí el filtro de imágenes por extensión, agregué alias `Repo`, docstrings y validaciones de entradas.
- `Code/adapters/__init__.py`: re-exporta `Repo` y `Repositorio`.
- `Code/vision/KMeansModel.py`: se agregó `fit` como API principal, parada por tolerancia explícita, manejo de semillas reproducibles, centroides en `float32` y recomputo final consistente.
- `Code/vision/ImgOrchestrator.py` y `Code/vision/__init__.py`: se reescribió la orquestación con funciones de alto nivel (`fit_from_paths`, `identify*`, etc.), se creó un `ImgOrchestrator` liviano compatible con la API anterior y se documentó el flujo.
- `Code/audio/AudioPreproc.py`: nuevo método `process_path` con carga segura de WAV y conversión a `float32`.
- `Code/audio/AudioOrchestrator.py` y `Code/audio/__init__.py`: implementación completa del orquestador de audio (construcción, persistencia y predicción batch) + exponer funciones de conveniencia.
- `Code/Statistics/BayesAgent.py`: cálculo log-estable usando `logsumexp`, control de tolerancias y retornos en `float32`.
- `Code/tests/`: nueva batería `pytest` para visión, audio, Bayes y repositorio.

# Errores detectados y correcciones
- `Code/adapters/Repositorio.py:47`: el filtro de imágenes comparaba la carpeta con las extensiones válidas; ahora se valida realmente `Path.suffix`.
- `Code/vision/KMeansModel.py`: la parada dependía simultáneamente de desplazamiento y cambio de etiquetas, lo que podía ciclar; ahora se usa la norma del desplazamiento y se recalculan etiquetas con los centroides finales.
- `Code/Statistics/BayesAgent.py:66`: la normalización en log-space no empleaba un `logsumexp` robusto; se sustituyó por `_logsumexp` con casting a `float64` previo.

# Mejores prácticas sugeridas (no aplicadas)
- Añadir integración de logging estructurado para orquestadores y repo (se pospuso para mantener el alcance solicitado).
- Incorporar validación extensiva de parámetros en `audio/KnnModel.py` (requiere revisar datasets reales).

# Pruebas y ejecución
```
export PYTHONPATH=./Code
pytest -q
```
