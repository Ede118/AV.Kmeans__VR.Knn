# AV.Kmeans__VR.Knn

### Proyecto Final – Inteligencia Artificial I

**Facultad de Ingeniería – Universidad Nacional de Cuyo**

---

## Descripción General

Este proyecto implementa un **sistema multiagente** que integra **visión artificial**, **reconocimiento de voz** y **razonamiento bayesiano** para resolver un problema de clasificación e inferencia probabilística en un entorno controlado.

El sistema toma una **muestra de 10 piezas metálicas** (tornillos, tuercas, arandelas y clavos) y, a partir de una secuencia de imágenes, determina la proporción de cada tipo.  
Luego, mediante inferencia bayesiana, estima **cuál de las cajas (A, B, C o D)** fue la fuente de las piezas, basándose en distribuciones previamente conocidas.

El usuario interactúa por **voz**, utilizando tres comandos reconocidos por el sistema:

- 🎙️ `"contar"` → muestra el conteo de piezas detectadas.
    
- 🎙️ `"proporción"` → ejecuta la inferencia bayesiana.
    
- 🎙️ `"salir"` → finaliza la ejecución.
    

---

## Arquitectura del Sistema

El proyecto se organiza modularmente en cinco subsistemas principales bajo el directorio `Code/`:

```
Code/
 ┣ app/
 ┃ ┣ AppController.py
 ┃ ┗ AC_pruebas.ipynb
 ┣ audio/
 ┃ ┣ AudioPreproc.py
 ┃ ┣ AudioFeat.py
 ┃ ┣ Standardizer.py
 ┃ ┣ KnnModel.py
 ┃ ┗ AudioOrchestrator.py
 ┣ Estadisticas/
 ┃ ┣ BayesAgent.py
 ┃ ┗ BayesTest.ipynb
 ┣ image/
 ┃ ┣ ImgPreproc.py
 ┃ ┣ KmeansModel.py
 ┃ ┗ Standardizer.py
 ┗ ui/
    ┣ AliasesUsed.py
    ┗ main.py
```

Además, contiene los directorios:

- `Database/` → bases de datos de imágenes y audios (`data/`, `input/`, `models/`).
    
- `Docs/` → documentación técnica, diagramas (Draw.io / Umbrello) y consignas.
    

---

## Módulos del Sistema

### 🔹 1. Visión Artificial (`image/`)

- **Objetivo:** clasificar imágenes de piezas metálicas en cuatro categorías.
    
- **Algoritmo:** `K-Means` implementado **desde cero** con `NumPy`.
    
- **Características extraídas:**
    
    - Cantidad de huecos ($\chi$)
        
    - Rugosidad del borde ($r_{\text{hull}}$)
        
    - Variación radial ($r_{\text{var}}$)
        
- **Salida:** vector de conteos $\mathbf{n} = [n_1, n_2, n_3, n_4]$.
    
- **Notebook de validación:** `ArtificialVision.ipynb`.
    

📸 _El entorno de captura está diseñado con iluminación difusa y fondo blanco, asegurando uniformidad en la segmentación._

---

### 🔹 2. Reconocimiento de Voz (`audio/`)

- **Objetivo:** detectar comandos hablados.
    
- **Algoritmo:** `K-Nearest Neighbors (KNN)` implementado manualmente en `numpy`.
    
- **Características:**
    
    - MFCC (Mel-Frequency Cepstral Coefficients)
        
    - Energía promedio
        
    - Zero-Crossing Rate (ZCR)
        
- **Preprocesamiento:**
    
    - Conversión a mono y resampleo (16 kHz)
        
    - Pre-énfasis
        
    - Detección de actividad de voz (VAD)
        
    - Normalización de amplitud
        
    - Ventana fija de duración
        
- **Salida:** clase de comando (`contar`, `proporción`, `salir`)
    
- **Notebook:** `VoiceRecognition.ipynb`.
    

🎧 _Validado con esquema “leave-one-speaker-out” (LOSO) para medir robustez ante nuevos hablantes._

---

### 🔹 3. Agente Bayesiano (`Estadisticas/`)

- **Objetivo:** inferir cuál de las cajas fue abierta, dadas las proporciones observadas.
    
- **Modelo probabilístico:**  
    [
    P(H_k \mid \mathbf{n}) =  
    \frac{P(H_k)\prod_i p_{k,i}^{n_i}}{\sum_j P(H_j)\prod_i p_{j,i}^{n_i}}  
    ]
    
- **Implementación estable numéricamente:**  
    $$  
    s_k = \log P(H_k) + \sum_i n_i\log(p_{k,i}+\varepsilon), \quad  
    P(H_k\mid\mathbf{n}) = \text{softmax}(s_k)  
    $$
    
- **Salida:**
    
    - Hipótesis más probable $H^*$
        
    - Vector de probabilidades posteriores $[P(H_A), P(H_B), P(H_C), P(H_D)]$
        
- **Notebook:** `BayesTest.ipynb`.
    

_La inferencia se activa al reconocer el comando `"proporción"`, orquestado por `AppController.py`._

---

### 🔹 4. Controlador de Aplicación (`app/AppController.py`)

Integra los tres agentes principales (visión, voz y bayesiano) en una única interfaz.  
Gestiona la secuencia de tareas, la comunicación entre módulos y la respuesta del sistema.

---

### 🔹 5. Interfaz de Usuario (`ui/`)

- Define alias y rutinas de salida visual (`AliasesUsed.py`, `main.py`).
    
- Provee una interfaz simple CLI o ventana para interacción directa.
    

---

## Flujo de Ejecución

1. **Captura y clasificación visual**
    
    - Se procesan 10 imágenes → conteo de clases $\mathbf{n}$.
        
2. **Reconocimiento de voz**
    
    - Se graba un comando y se clasifica con `KNN`.
        
3. **Inferencia bayesiana**
    
    - Si el comando es `"proporción"`, se ejecuta el cálculo posterior.
        
4. **Visualización de resultados**
    
    - Se imprime la caja más probable y la distribución de probabilidades.
        

---

## Resultados Esperados

|Módulo|Métrica|Objetivo|Estado|
|---|---|---|---|
|Visión Artificial|Macro Accuracy ≥ 85%|✅ Validado||
|Reconocimiento de Voz|LOSO Accuracy ≥ 85%|✅ Validado||
|Agente Bayesiano|Acierto ≥ 95% (simulado)|✅ Validado||

---

## Tecnologías y Librerías

|Componente|Librerías utilizadas|
|---|---|
|Visión|`opencv-python`, `numpy`, `matplotlib`, `scipy`|
|Audio|`librosa`, `sounddevice`, `numpy`, `pandas`|
|Bayes|`numpy`, `math`, `matplotlib`|
|General|`jupyter`, `time`, `logging`, `os`, `pathlib`|

> ⚠️ **No se utiliza Scikit-learn.**  
> Todos los algoritmos (`K-Means`, `KNN`, inferencia Bayesiana) fueron implementados de forma **manual y reproducible** con `NumPy`.

---

## Ejecución

Referirse al PDF correspondiente.

---

## Créditos y Referencias

- Consigna oficial del **Trabajo Final – Inteligencia Artificial I** (UNCuyo, 2025).
- Documentación técnica incluida en `/Docs/`.
- Librerías oficiales:
    - [NumPy](https://numpy.org/doc/)
    - [OpenCV](https://docs.opencv.org/)
    - [Librosa](https://librosa.org/doc/)
    - [Matplotlib](https://matplotlib.org/stable/contents.html)

