from __future__ import annotations

import warnings
from typing import List, Optional, Sequence, Tuple

import numpy as np
from dataclasses import dataclass, field

@dataclass(slots=True)
class KMeansModel:

    n_clusters: int = 2
    max_iter: int = 300
    tol: float = 1e-4

    init_centers: np.ndarray | None = None
    random_state : int | None = None

    centers_: np.ndarray | None = None
    inertia_: float | None = None

    def __post_init__(self) -> None:
        if self.n_clusters <= 0:
            raise ValueError(f"Cantidad de clusters debe ser mayor a cero: {self.n_cluster}")
        if self.max_iter <= 0:
            raise ValueError(f"Cantidad de iteraciones debe ser mayor a cero: {self.max_iter}")
        
    def fit(self, X: np.ndarray, init_centers: np.ndarray | None = None) -> np.ndarray:
        X = self._check_X(X)
        F = X.shape[1]
        centers0 = init_centers if init_centers is not None else self.init_centers
        if centers0 is not None:
            centers = np.asarray(centers0, dtype=np.float64)
            if centers.shape != (self.n_clusters, F):
                raise ValueError(f"init_centers debe tener forma ({self.n_clusters}, {F})")
        else:
            centers = self._init_centers(X)

        for _ in range(self.max_iter):
            labels = self._assign_labels(X, centers)
            new_centers = self._update_centers(X, labels, centers)
            shift = np.linalg.norm(new_centers - centers, axis=1).max()
            centers = new_centers
            if shift < self.tol:
                break

        self.centers_ = centers
        diff = X - centers[labels]
        self.inertia_ = float(np.sum(diff**2))
        return centers

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.centers_ is None:
            raise RuntimeError("Debes llamar a fit(X) antes de predict(X).")
        X = self._check_X(X)
        labels = self._assign_labels(X, self.centers_)
        return labels

    # ------------------------------------------------------------------ #
    def _check_X(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X debe ser un array 2D de forma (N, F).")
        N, F = X.shape
        if N < self.n_clusters:
            raise ValueError(f"n_clusters={self.n_clusters} no puede ser mayor que N={N}.")
        return X

    def _init_centers(self, X: np.ndarray) -> np.ndarray:
        N, F = X.shape
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(N, size=self.n_clusters, replace=False)
        centers = X[idx, :].copy()
        return centers

    def _assign_labels(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """
        Calcula distancias cuadradas de cada punto a cada centro:
          d[i, k] = || X[i] - centers[k] ||^2
        y asigna el centro más cercano.
        """
        # X: (N, F), centers: (K, F)
        # Usamos broadcasting: (N, 1, F) - (1, K, F) -> (N, K, F)
        diff = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)  # (N, K)
        labels = np.argmin(dist_sq, axis=1)  # (N,)
        return labels

    def _update_centers(self, X: np.ndarray, labels: np.ndarray, old_centers: np.ndarray) -> np.ndarray:
        N, F = X.shape
        K = self.n_clusters
        centers = np.zeros((K, F), dtype=np.float64)

        for k in range(K):
            mask = (labels == k)
            if not np.any(mask):
                # Cluster vacío: re-inicializar en un punto aleatorio
                idx = np.random.randint(0, N)
                centers[k] = X[idx]
            else:
                centers[k] = X[mask].mean(axis=0)

        return centers
