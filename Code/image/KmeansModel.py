from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from Code.types import MatF, VecF, VecI


@dataclass(slots=True)
class KMeans:
    """
    ### Explicación
    Extractor de características geométricas/fotométricas para objetos segmentados.
    
    Usa la imagen normalizada y la máscara asociada (producidas por `ImgPreproc`)
    para construir vectores de features que alimentan modelos de clustering o clasificación.
    
    ### Métodos principales:
    
    - `shape_vector`: arma un descriptor compacto (3D o 5D) con huecos, circularidad,
      vértices, rugosidad y gradiente.
    - `contar_huecos`, `contar_vertices`, `circularidad`, `rugosidad`, `gradiente_interno`:
      calculan métricas atómicas reutilizables en otros pipelines.
    - `feature_names`: devuelve etiquetas legibles para cada componente del descriptor.
    
    Las entradas (`img_norm`, `mask`) deben estar alineadas y en escala 0..1 (imagen)
    y {0,1}/{0,255} (máscara). Cada helper robustifica la máscara para tolerar ambos rangos.
    """

    k: int
    tol: float = 1e-4
    max_iter: int = 100
    random_state: Optional[int] = None

    matCentroides_: Optional[MatF] = field(default=None, init=False, repr=False)
    vecLabels_: Optional[VecI] = field(default=None, init=False, repr=False)
    inercia_: Optional[float] = field(default=None, init=False, repr=False)
    _rng: np.random.Generator = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.k <= 0:
            raise ValueError("k debe ser > 0")
        self._rng = np.random.default_rng(self.random_state)

    # --------- API pública ---------
    def fit(
        self, 
        matParametros: MatF, 
        seeds: Optional[MatF] = None
        ) -> "KMeans":
        """
        ### Entrenamiento del modelo
        Ajusta los centroides de K-Means sobre `matParametros`.
        - Acepta semillas opcionales (`seeds`) con shape (k, D)
        - Inicializa determinísticamente si `random_state` está definido
        - Itera hasta que el desplazamiento máximo de centroides cae por debajo de `tol`
        ### Resumen

        ```
        modelo = KMeans(k=3, random_state=0)
        modelo.fit(matParametros)
        ```
        """

        X = np.asarray(matParametros, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("matParametros debe tener shape (N, D)")
        n_samples, dim = X.shape
        if n_samples < self.k:
            raise ValueError("k no puede ser mayor que la cantidad de muestras.")

        if self.random_state is not None:
            self._rng = np.random.default_rng(self.random_state)
        elif self._rng is None:
            self._rng = np.random.default_rng()

        if seeds is not None:
            C = np.asarray(seeds, dtype=np.float64)
            if C.shape != (self.k, dim):
                raise ValueError(f"seeds debe tener shape {(self.k, dim)}, no {C.shape}")
        else:
            indices = self._rng.choice(n_samples, size=self.k, replace=False)
            C = X[indices].copy()

        previous_labels: Optional[np.ndarray] = None
        stable_iters = 0
        
        for _ in range(self.max_iter):
            dist2 = self._distancias_a_centroides(X, C)
            labels, _ = self._determinar_labels(dist2)
            C_new, shift = self._actualizar_centroides(X, labels, C)
            converged = shift <= self.tol
            C = C_new
            
            if np.array_equal(labels, previous_labels):
                stable_iters += 1
            else:
                stable_iters = 0

            if converged or stable_iters > 2:  # 3 iteraciones extra
                break
            
            previous_labels = labels

        # Asegurar consistencia final
        dist2_final = self._distancias_a_centroides(X, C)
        labels_final, d2 = self._determinar_labels(dist2_final)

        self.matCentroides_ = C.astype(np.float64, copy=False)
        self.vecLabels_ = labels_final.astype(np.int64, copy=False)
        self.inercia_ = float(np.sum(d2, dtype=np.float64))
        return self


    def predict(
        self, 
        matParametros: MatF
        ) -> VecI:
        """
        ### Predicción de clusters
        Asigna cada fila de `matParametros` al centroide más cercano.
        - Requiere haber entrenado (`matCentroides_` no puede ser None)
        - Devuelve labels `int64` con shape (N,)
        ### Resumen

        ```
        labels = modelo.predict(matParametros_nuevos)
        ```
        """
        
        if self.matCentroides_ is None:
            raise RuntimeError("Modelo no entrenado. Llamá a fit() primero.")
        X = np.asarray(matParametros, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("matParametros debe tener shape (N, D)")
        if X.shape[1] != self.matCentroides_.shape[1]:
            raise ValueError(
                f"Dimensión de entrada {X.shape[1]} incompatible con centroides {self.matCentroides_.shape[1]}"
            )
        dist2 = self._distancias_a_centroides(X, self.matCentroides_)
        labels, _ = self._determinar_labels(dist2)
        return labels.astype(np.int64, copy=False)

    def predict_info(
        self, 
        matParametros: MatF
        ) -> Tuple[VecI, VecF]:
        """
        ### Predicción con distancias
        Devuelve labels y distancia cuadrática mínima para cada vector.
        - Usa los centroides entrenados
        - Retorna `(labels, distancias²)` ambos en float32/int64
        ### Resumen

        ```
        labels, dist2 = modelo.predict_info(matParametros_nuevos)
        ```
        """
        
        if self.matCentroides_ is None:
            raise RuntimeError("Modelo no entrenado. Llamá a fit() primero.")
        X = np.asarray(matParametros, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("matParametros debe tener shape (N, D)")
        if X.shape[1] != self.matCentroides_.shape[1]:
            raise ValueError(
                f"Dimensión de entrada {X.shape[1]} incompatible con centroides {self.matCentroides_.shape[1]}"
            )
        dist2 = self._distancias_a_centroides(X, self.matCentroides_)
        labels, d2 = self._determinar_labels(dist2)
        return labels.astype(np.int64, copy=False), d2.astype(np.float64, copy=False)

    # --------- Núcleo privado ---------
    @staticmethod
    def _distancias_a_centroides(
        matParametros: MatF, 
        matCentroides: MatF
        ) -> np.ndarray:
        """
        ### Distancias cuadráticas
        Calcula la matriz `D[i,j] = ||x_i - c_j||²` en float32 estable.
        - Acepta matrices convertibles a float32
        - Usa expansión vectorizada con productos y normas
        ### Resumen

        ```
        dist2 = KMeans._distancia_centroide(X, C)
        ```
        """
        
        X = np.asarray(matParametros, dtype=np.float64)
        C = np.asarray(matCentroides, dtype=np.float64)
        x2 = np.sum(X * X, axis=1, keepdims=True, dtype=np.float64)
        c2 = np.sum(C * C, axis=1, keepdims=True, dtype=np.float64).T
        prod = X @ C.T
        dist2 = np.maximum(x2 + c2 - 2.0 * prod, 0.0)
        return dist2.astype(np.float64, copy=False)

    @staticmethod
    def _determinar_labels(
        matDist2: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        ### Etiquetado óptimo
        Selecciona el centroide más cercano y su distancia por fila.
        - Usa `argmin` sobre axis=1
        - Retorna `(labels, dist_min)` compatibles con float32/int64
        ### Resumen

        ```
        labels, d2 = KMeans._asignar(dist2)
        ```
        """
        labels = np.argmin(matDist2, axis=1).astype(np.int64)
        d2_min = matDist2[np.arange(matDist2.shape[0]), labels].astype(np.float64)
        return labels, d2_min

    def _actualizar_centroides(
        self,
        matParametros: MatF,
        vecLabels: VecI,
        matCentroidesPrev: Optional[MatF] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        ### Re-cálculo de centroides
        Promedia los puntos por cluster y mide el desplazamiento máximo.
        - Calcula nuevos centroides o reutiliza previos cuando un cluster queda vacío
        - Devuelve `(centroides, shift)` donde `shift` es el mayor movimiento L2
        ### Resumen

        ```
        C_new, shift = modelo._calcular_centroides(X, labels, C_prev)
        ```
        """
        
        X = np.asarray(matParametros, dtype=np.float64)
        y = np.asarray(vecLabels, dtype=np.int64)
        k = int(self.k)
        dim = X.shape[1]

        centroids = np.empty((k, dim), dtype=np.float64)
        for idx in range(k):
            mask = y == idx
            if np.any(mask):
                centroids[idx] = X[mask].mean(axis=0, dtype=np.float64)
            elif matCentroidesPrev is not None:
                centroids[idx] = np.asarray(matCentroidesPrev, dtype=np.float64)[idx]
            else:
                mu = X.mean(axis=0, dtype=np.float64)
                dist2 = np.sum((X - mu) ** 2, axis=1)
                centroids[idx] = X[int(np.argmax(dist2))]

        shift = 0.0
        if matCentroidesPrev is not None:
            diff = centroids - np.asarray(matCentroidesPrev, dtype=np.float64)
            shift = float(np.linalg.norm(diff, axis=1).max(initial=0.0))
        else:
            shift = float("inf")
        return centroids, shift

    def _clasificar_vector(
        self, 
        vecX: VecF, 
        matCentroides: MatF
    ) -> Tuple[int, float, int]:
        """
        ### Clasificación puntual
        Evalúa un vector individual frente a los centroides dados.
        - Calcula distancias cuadráticas y obtiene label ganador
        - Retorna `(label, distancia², segundo_mejor)` para análisis adicional
        ### Resumen

        ```
        label, d2, segundo = modelo._clasificar_elemento(x, centroides)
        ```
        """
        
        
        x = np.asarray(vecX, dtype=np.float64).reshape(1, -1)
        centroides = np.asarray(matCentroides, dtype=np.float64)
        if centroides.ndim != 2:
            raise ValueError("matCentroides debe tener shape (k, D)")
        if x.shape[1] != centroides.shape[1]:
            raise ValueError(
                f"Dimensión del vector {x.shape[1]} incompatible con centroides {centroides.shape[1]}"
            )
        
        dist2 = self._distancias_a_centroides(x, centroides)
        
        labels, d2_min = self._determinar_labels(dist2)
        order = np.argsort(dist2[0])
        second = int(order[1]) if centroides.shape[0] > 1 else -1
        return int(labels[0]), float(d2_min[0]), second

    # Alias públicos en español e histórico
    def entrenar_clusters(
        self,
        matParametros: MatF,
        seeds: Optional[MatF] = None
    ) -> "KMeans":
        return self.fit(matParametros, seeds=seeds)

    def clasificar_vector(
        self,
        matParametros: MatF
    ) -> VecI:
        return self.predict(matParametros)

    def asignar_con_distancias(
        self,
        matParametros: MatF
    ) -> Tuple[VecI, VecF]:
        return self.predict_info(matParametros)

    def ajustar(
        self,
        matParametros: MatF,
        semillas: Optional[MatF] = None
    ) -> "KMeans":
        return self.fit(matParametros, seeds=semillas)

    def predecir_indices(
        self,
        matParametros: MatF
    ) -> VecI:
        return self.predict(matParametros)

    def predecir_con_distancias(
        self,
        matParametros: MatF
    ) -> Tuple[VecI, VecF]:
        return self.predict_info(matParametros)

    # Alias privados para compatibilidad
    @staticmethod
    def _distancia_centroide(matParametros: MatF, matCentroides: MatF) -> np.ndarray:
        return KMeans._distancias_a_centroides(matParametros, matCentroides)

    @staticmethod
    def _asignar(matDist2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return KMeans._determinar_labels(matDist2)

    def _calcular_centroides(
        self,
        matParametros: MatF,
        vecLabels: VecI,
        matCentroidesPrev: Optional[MatF] = None
    ) -> Tuple[np.ndarray, float]:
        return self._actualizar_centroides(matParametros, vecLabels, matCentroidesPrev)

    def _clasificar_elemento(
        self,
        vecX: VecF,
        matCentroides: MatF
    ) -> Tuple[int, float, int]:
        return self._clasificar_vector(vecX, matCentroides)

    @staticmethod
    def _asignar_etiquetas(matDist2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return KMeans._determinar_labels(matDist2)

    def _recalcular_centroides(
        self,
        matParametros: MatF,
        vecLabels: VecI,
        matCentroidesPrev: Optional[MatF] = None,
    ) -> Tuple[np.ndarray, float]:
        return self._actualizar_centroides(matParametros, vecLabels, matCentroidesPrev)

    def _inercia(self) -> Optional[float]:
        return self.inercia_
