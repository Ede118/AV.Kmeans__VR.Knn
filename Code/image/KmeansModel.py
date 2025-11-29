from __future__ import annotations

import warnings
from typing import List, Optional, Sequence, Tuple

import numpy as np
from sklearn.cluster import KMeans as SKKMeans
from sklearn.preprocessing import StandardScaler


class KMeansModel:
    """Wrapper fino sobre StandardScaler + sklearn.KMeans con trazabilidad de features."""

    def __init__(self, n_clusters: int = 4, random_state: Optional[int] = 0) -> None:
        if n_clusters <= 0:
            raise ValueError("n_clusters debe ser > 0")
        self.n_clusters = int(n_clusters)
        self.random_state = random_state
        self.scaler_: Optional[StandardScaler] = None
        self.model_: Optional[SKKMeans] = None
        self.feature_names_: Optional[List[str]] = None
        self.dim_: Optional[int] = None

    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray, feature_names: Optional[Sequence[str]] = None) -> "KMeansModel":
        """Ajusta scaler + KMeans y registra dimensión/feature names."""

        data = np.asarray(X, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError("X debe tener shape (N, D)")
        n_samples, dim = data.shape
        if n_samples < self.n_clusters:
            raise ValueError("n_clusters no puede superar la cantidad de muestras.")

        self.dim_ = dim
        if feature_names is not None:
            feature_names = list(feature_names)
            if len(feature_names) != dim:
                raise ValueError(f"feature_names debe tener longitud {dim}, recibió {len(feature_names)}")
            self.feature_names_ = list(feature_names)
        else:
            self.feature_names_ = [f"f{i}" for i in range(dim)]

        self.scaler_ = StandardScaler()
        data_scaled = self.scaler_.fit_transform(data)

        self.model_ = SKKMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init="auto",
        )
        self.model_.fit(data_scaled)
        return self

    # ------------------------------------------------------------------ #
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predice labels validando que la dimensión coincida con la usada en fit."""

        self._assert_fitted()
        data = np.asarray(X, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError("X debe tener shape (N, D)")
        if data.shape[1] != self.dim_:
            raise ValueError(
                f"Se esperaban vectores con D={self.dim_} (features={self.feature_names_}), "
                f"pero se recibió shape {data.shape}"
            )
        data_scaled = self.scaler_.transform(data)
        labels = self.model_.predict(data_scaled)
        return labels.astype(np.int64, copy=False)

    # ------------------------------------------------------------------ #
    def predict_one(self, vec: np.ndarray) -> int:
        """Atajo para predecir una sola muestra."""

        sample = np.asarray(vec, dtype=np.float64)
        if sample.ndim == 1:
            sample = sample.reshape(1, -1)
        elif sample.ndim != 2 or sample.shape[0] != 1:
            raise ValueError("predict_one acepta un vector (D,) o (1, D)")
        return int(self.predict(sample)[0])

    # ------------------------------------------------------------------ #
    def fit_predict(
        self,
        X: np.ndarray,
        feature_names: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        """Ajusta el modelo y devuelve los labels en un único paso."""

        return self.fit(X, feature_names=feature_names).predict(X)

    # ------------------------------------------------------------------ #
    def get_centers(self) -> Tuple[np.ndarray, Optional[List[dict]]]:
        """
        Retorna centroides desescalados y, si hay nombres, la versión mapeada.

        Returns
        -------
        centers : np.ndarray
            Centroides en el espacio original (shape (k, D)).
        named_centers : list[dict] | None
            Lista de diccionarios {feature: valor} por centroide.
        """

        self._assert_fitted()
        centers_scaled = self.model_.cluster_centers_
        centers = self.scaler_.inverse_transform(centers_scaled)
        named = None
        if self.feature_names_:
            named = [
                {name: float(val) for name, val in zip(self.feature_names_, center)}
                for center in centers
            ]
        return centers.astype(np.float64, copy=False), named

    # ------------------------------------------------------------------ #
    def _assert_fitted(self) -> None:
        if self.model_ is None or self.scaler_ is None or self.dim_ is None:
            raise RuntimeError("El modelo no está entrenado. Llamá primero a fit().")


# ------------------------------------------------------------------------------ #
# Legacy wrapper for backwards compatibility
# ------------------------------------------------------------------------------ #
class KMeans(KMeansModel):
    """Compatibilidad con la API anterior basada en KMeans propio."""

    def __init__(
        self,
        k: int,
        tol: float = 1e-4,
        max_iter: int = 100,
        random_state: Optional[int] = None,
    ) -> None:
        warnings.warn(
            "KMeans está deprecado. Usa KMeansModel para obtener scaler y validaciones.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(n_clusters=k, random_state=random_state)
        self._legacy_tol = tol
        self._legacy_max_iter = max_iter

    def fit(self, matParametros: np.ndarray, seeds: Optional[np.ndarray] = None) -> "KMeans":
        if seeds is not None:
            warnings.warn(
                "El parámetro seeds se ignora en la versión basada en sklearn.",
                DeprecationWarning,
                stacklevel=2,
            )
        super().fit(matParametros)
        return self

    def predict(self, matParametros: np.ndarray) -> np.ndarray:
        return super().predict(matParametros)

    def predict_one(self, vec: np.ndarray) -> int:
        return super().predict_one(vec)

    def fit_predict(self, matParametros: np.ndarray) -> np.ndarray:
        return super().fit_predict(matParametros)

    def predict_info(self, matParametros: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        warnings.warn(
            "predict_info está deprecado; usa KMeansModel.get_centers para analizar distancias.",
            DeprecationWarning,
            stacklevel=2,
        )
        labels = self.predict(matParametros)
        data = np.asarray(matParametros, dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        data_scaled = self.scaler_.transform(data)
        dist = self.model_.transform(data_scaled)
        min_dist = np.min(dist, axis=1)
        return labels, (min_dist ** 2).astype(np.float64, copy=False)
