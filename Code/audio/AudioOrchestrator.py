from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from Code.audio.AudioFeat import AudioFeat
from Code.audio.AudioPreproc import AudioPreproc
from Code.audio.Standardizer import Standardizer
from Code.audio.KnnModel import KnnModel, KnnConfig

AudioPath = str | Path


@dataclass(slots=True)
class AudioOrchestrator:
    """
    Coordina preprocesamiento, extracción de features, PCA opcional y k-NN.
    - `entrenar` arma la base y deja el KNN listo (y puede guardar el modelo).
    - `cargar_modelo` restaura lo guardado.
    - `predecir_comando` aplica el pipeline de inferencia.
    """

    preproc: AudioPreproc = field(default_factory=AudioPreproc)
    feat: AudioFeat = field(default_factory=AudioFeat)
    stats: Standardizer = field(default_factory=Standardizer)
    knn: KnnModel = field(default_factory=KnnModel)
    _X_store: np.ndarray | None = field(default=None, init=False, repr=False)       # (N, D) std
    _y_store: np.ndarray | None = field(default=None, init=False, repr=False)       # (N,)
    _X_store_raw: np.ndarray | None = field(default=None, init=False, repr=False)   # (N, D_raw)
    _X_store_proj: np.ndarray | None = field(default=None, init=False, repr=False)  # (N, k)
    _feature_names: list[str] | None = field(default=None, init=False, repr=False)
    _eigvecs: np.ndarray | None = field(default=None, init=False, repr=False)       # (D, D)
    _eigvals: np.ndarray | None = field(default=None, init=False, repr=False)       # (D,)
    _k_used: int | None = field(default=None, init=False, repr=False)

    # -------------------------------------------------------------------------------------------------  #
    #                                      --------- Entrenamiento ---------                             #
    # -------------------------------------------------------------------------------------------------  #

    def entrenar(
        self,
        paths: Sequence[AudioPath],
        labels: Sequence[str],
        *,
        var_objetivo: float = 0.95,
        k_componentes: int | None = None,
        guardar_en: Path | str | None = None,
    ) -> dict[str, int | float]:
        """
        ### Entrenar orquestador
        Preprocesa audios, extrae features, ajusta Standardizer, aplica PCA y carga KNN.
        - `var_objetivo`: fracción de varianza a retener si no fijas `k_componentes`
        - `k_componentes`: fija explícitamente cuántas PCs usar (ignora `var_objetivo`)
        - `guardar_en`: ruta opcional para persistir el modelo al final
        """
        path_list = list(paths)
        label_list = [str(l) for l in labels]
        if not path_list:
            raise ValueError("paths no puede estar vacío.")
        if len(path_list) != len(label_list):
            raise ValueError("paths y labels deben tener la misma longitud.")

        # 1) Preproceso + features
        vectores_raw: list[np.ndarray] = []
        for p in path_list:
            y_proc, sr = self.preproc.preprocesar_desde_path(p)
            vec = self.feat.extraer_caracteristicas(y_proc, sr)
            vectores_raw.append(vec.astype(np.float64, copy=False))

        X_raw = np.stack(vectores_raw, axis=0).astype(np.float64, copy=False)  # (N, D_raw)
        self._feature_names = self.feat.nombres_features()

        # 2) Standardizer (Z-score)
        self.stats.calculate_statistics(X_raw)
        X_std = self.stats.transform(X_raw).astype(np.float32, copy=False)

        # 3) PCA sobre X_std
        eigvals, eigvecs = self._pca(X_std)
        if k_componentes is None:
            k = int(np.searchsorted(np.cumsum(eigvals) / eigvals.sum(), float(var_objetivo)) + 1)
        else:
            k = max(1, min(int(k_componentes), eigvecs.shape[1]))
        
        X_proj = X_std @ eigvecs[:, :k]

        # 4) KNN con proyecciones
        self.knn.cargar_lote(X_proj.astype(np.float32, copy=False), label_list)

        # 5) Persistir estado en memoria
        self._X_store_raw = X_raw
        self._X_store = X_std
        self._X_store_proj = X_proj
        self._y_store = np.asarray(label_list, dtype=np.str_)
        self._eigvecs = eigvecs
        self._eigvals = eigvals
        self._k_used = k

        if guardar_en is not None:
            self.guardar_modelo(guardar_en)

        return {
            "N": X_raw.shape[0],
            "D_raw": X_raw.shape[1],
            "D_proj": k,
            "var_retenida": float(eigvals[:k].sum() / eigvals.sum()),
        }

    # -------------------------------------------------------------------------------------------------  #
    #                                         --------- Modelo ---------                                 #
    # -------------------------------------------------------------------------------------------------  #

    def guardar_modelo(self, path: Path | str) -> None:
        """
        ### Guardar modelo entrenado
        Persiste stats, PCA y base proyectada para reconstruir el KNN luego.
        """
        self._ensure_listo()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            mu=self.stats.mu,
            sigma=self.stats.sigma,
            eigvals=self._eigvals,
            eigvecs=self._eigvecs,
            k_used=self._k_used,
            X_proj=self._X_store_proj,
            labels=np.array(self._y_store, dtype=np.str_),
            feature_names=np.array(self._feature_names, dtype=object) if self._feature_names else None,
            knn_k=self.knn.cfg.k,
            knn_metric=str(self.knn.cfg.tipo_distancia),
            knn_weighted=self.knn.cfg.weighted,
        )

    def cargar_modelo(self, path: Path | str) -> None:
        """
        ### Cargar modelo
        Restaura Standardizer, PCA y base proyectada; reconstruye KNN.
        """
        data = np.load(path, allow_pickle=False)

        mu = data["mu"].astype(np.float32, copy=False)
        sigma = data["sigma"].astype(np.float32, copy=False)
        eigvals = data["eigvals"].astype(np.float64, copy=False)
        eigvecs = data["eigvecs"].astype(np.float64, copy=False)
        k_used = int(data["k_used"])
        X_proj = data["X_proj"].astype(np.float32, copy=False)
        labels = data["labels"].astype(str)
        feature_names = data["feature_names"]
        knn_k = int(data["knn_k"])
        knn_metric_raw = data["knn_metric"]
        knn_metric = str(knn_metric_raw.item() if hasattr(knn_metric_raw, "item") else knn_metric_raw)
        if knn_metric not in ("cosine", "euclidean"):
            raise ValueError(f"Métrica KNN inválida cargada: {knn_metric}")
        knn_weighted = bool(data["knn_weighted"])

        self.stats.mu = mu
        self.stats.sigma = sigma
        self._eigvals = eigvals
        self._eigvecs = eigvecs
        self._k_used = k_used
        self._feature_names = list(feature_names.tolist())
        self._X_store_proj = X_proj
        self._X_store = None
        self._X_store_raw = None
        self._y_store = labels

        # reconstruir KNN con config guardada
        self.knn = KnnModel(cfg=KnnConfig(k=knn_k, tipo_distancia=knn_metric, weighted=knn_weighted))
        self.knn.cargar_lote(X_proj, labels.tolist())

    # -------------------------------------------------------------------------------------------------  #
    #                                        --------- Predicción ---------                              #
    # -------------------------------------------------------------------------------------------------  #

    def predecir_comando(
        self,
        entrada: AudioPath | tuple[np.ndarray, int] | np.ndarray,
        *,
        devolver_distancia: bool = True,
    ) -> dict[str, str | float]:
        """
        ### Inferencia
        Preprocesa la entrada, extrae features, z-score, proyecta y predice con KNN.
        """
        self._ensure_listo()
        y_proc, sr = self._preprocesar_audio(entrada)
        vec_raw = self.feat.extraer_caracteristicas(y_proc, sr)
        vec_std = self.stats.transform_one(vec_raw)
        k = int(self._k_used or vec_std.shape[0])
        vec_proj = vec_std @ self._eigvecs[:, :k]

        label = self.knn.predecir(vec_proj.astype(np.float32, copy=False))
        salida: dict[str, str | float] = {"label": label}
        
        if devolver_distancia:
            distancias = self.knn.distancias(vec_proj.astype(np.float32, copy=False))
            salida["distancia_min"] = float(np.min(distancias))

        return salida

    # -------------------------------------------------------------------------------------------------  #
    #                                    --------- Helpers privados ---------                            #
    # -------------------------------------------------------------------------------------------------  #

    def _pca(self, X_std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        PCA clásico vía autovalores de la matriz de covarianza (X_std ya centrado/escala).
        Devuelve (eigvals_desc, eigvecs_ordenados).
        """
        cov = np.cov(X_std, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1]
        return eigvals[idx], eigvecs[:, idx]

    def _preprocesar_audio(
        self,
        entrada: AudioPath | tuple[np.ndarray, int] | np.ndarray,
    ) -> tuple[np.ndarray, int]:
        if isinstance(entrada, (str, Path)):
            return self.preproc.preprocesar_desde_path(entrada)
        if isinstance(entrada, tuple) and len(entrada) == 2:
            y, sr = entrada
            return self.preproc.procesar(np.asarray(y, dtype=np.float32), int(sr))
        if isinstance(entrada, np.ndarray):
            return self.preproc.procesar(np.asarray(entrada, dtype=np.float32), self.preproc.cfg.target_sr)
        raise TypeError("Entrada inválida. Usa path, (y, sr) o np.ndarray.")

    def _ensure_listo(self) -> None:
        if (
            self.stats.mu is None
            or self.stats.sigma is None
            or self._eigvecs is None
            or self._k_used is None
            or self.knn is None
        ):
            raise RuntimeError("Modelo no entrenado/cargado. Llamá a entrenar() o cargar_modelo().")


__all__ = ["AudioOrchestrator", "AudioPath"]
