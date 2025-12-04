from dataclasses import dataclass, field
import numpy as np
import scipy.signal as sps

from Code.types import VecF, MatF, F32

@dataclass(slots=True)
class Standardizer:
    """
    Z-score por dimensión: x' = (x - mu)/sigma.
    """
    mu: np.ndarray | None = None     # (D,)
    sigma: np.ndarray | None = None  # (D,)
    eps: float = 1e-8                   # evita divisiones por ~0

    # ---- entrenamiento (calcula estadísticas de TRAIN) ----
    def calculate_statistics(
        self, 
        X: np.ndarray
        ) -> "Standardizer":
        """
        ### Ajuste de estadísticas
        Calcula media y desvío por columna sobre una matriz `(N, D)`.
        - Almacena los valores internos en `float32`
        ### Resumen
        ```
        stats.calculate_statistics(X_train)
        ```
        """
        X = np.asarray(X, dtype=F32)
        
        if X.ndim != 2:
            raise ValueError("X debe ser 2D (N, D)")
        
        mu = X.mean(axis=0)
        sigma = X.std(axis=0)
        
        # columnas casi constantes → evita división por ~0
        sigma = np.where(sigma < self.eps, 1.0, sigma).astype(F32)
        
        self.mu = mu.astype(F32)
        self.sigma = sigma
        
        return self

    # ---- chequeo interno ----
    def _check(self) -> None:
        if self.mu is None or self.sigma is None:
            raise RuntimeError("Standardizer no está ajustado. Corre fit(X) primero.")

    # ---- transformación batch ----
    def transform(
        self, 
        X: MatF
        ) -> MatF:
        """
        ### Z-score por lote
        Estandariza arrays 1D o 2D usando las estadísticas almacenadas.
        - Devuelve resultado en `float32`
        ### Resumen
        ```
        X_std = stats.transform(X)
        ```
        """
        self._check()
        
        X = np.asarray(X, dtype=F32)
        
        if X.ndim == 1:
            return ((X - self.mu) / self.sigma).astype(F32)
        if X.ndim == 2:
            return ((X - self.mu[None, :]) / self.sigma[None, :]).astype(F32)
        
        raise ValueError("X debe ser 1D o 2D")

    # ---- transformación para un solo vector ----
    def transform_one(
        self, 
        x: VecF
        ) -> VecF:
        """
        ### Z-score vectorial
        Normaliza un único vector `(D,)` con mu y sigma ajustados.
        ### Resumen
        ```
        x_std = stats.transform_one(x)
        ```
        """

        self._check()
        x = np.asarray(x, dtype=F32)
        if x.ndim != 1:
            raise ValueError("x debe ser 1D (D,)")
        
        return ((x - self.mu) / self.sigma).astype(F32)

    # ---- inversa (útil para debug) ----
    def inverse_transform(
        self, 
        X: MatF
        ) -> MatF:
        """
        ### Deshacer Z-score
        Reconstruye valores originales desde datos estandarizados.
        - Compatible con entradas 1D o 2D
        ### Resumen
        ```
        X_orig = stats.inverse_transform(X_std)
        ```
        """
        self._check()
        X = np.asarray(X, dtype=F32)
        if X.ndim == 1:
            return (X * self.sigma + self.mu).astype(F32)
        if X.ndim == 2:
            return (X * self.sigma[None, :] + self.mu[None, :]).astype(F32)
        raise ValueError("X debe ser 1D o 2D")

    # ---- persistencia ----
    def save(
        self, 
        path: str
        ) -> None:
        """
        ### Guardar estadísticas
        Persiste mu y sigma en un archivo `.npz` comprimido.
        ### Resumen
        ```
        stats.save("audio_stats.npz")
        ```
        """
        self._check()
        np.savez_compressed(path, mu=self.mu, sigma=self.sigma)

    @classmethod
    def load(
        cls, 
        path: str
        ) -> "Standardizer":
        """
        ### Cargar estadísticas
        Restaura mu y sigma desde un `.npz` y devuelve una instancia lista.
        ### Resumen
        ```
        stats = Standardizer.load("audio_stats.npz")
        ```
        """
        data = np.load(path)
        mu = data["mu"].astype(F32)
        sigma = data["sigma"].astype(F32)
        return cls(mu=mu, sigma=sigma)

    # --- alias en español ---
    def ajustar_estadisticas(self, X: np.ndarray) -> "Standardizer":
        """
        ### Alias en español
        Delegación a `calculate_statistics`.
        ### Resumen
        ```
        stats.ajustar_estadisticas(X)
        ```
        """
        return self.calculate_statistics(X)

    def transformar(self, X: MatF) -> MatF:
        """
        ### Alias en español
        Delegación a `transform`.
        ### Resumen
        ```
        X_std = stats.transformar(X)
        ```
        """
        return self.transform(X)

    def transformar_uno(self, x: VecF) -> VecF:
        """
        ### Alias en español
        Delegación a `transform_one`.
        ### Resumen
        ```
        x_std = stats.transformar_uno(x)
        ```
        """
        return self.transform_one(x)
