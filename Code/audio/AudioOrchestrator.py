from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence, Union, List, Dict, Tuple

import numpy as np

from Code.audio.AudioFeat import AudioFeat
from Code.audio.AudioPreproc import AudioPreproc
from Code.audio.Standardizer import Standardizer
from Code.audio.KnnModel import KnnModel

if TYPE_CHECKING:
    from Code.adapters.Repositorio import Repo

AudioPath = str | Path


@dataclass(slots=True)
class AudioOrchestrator:
    """Coordinate audio preprocessing, feature extraction, and KNN inference."""

    preproc: AudioPreproc = field(default_factory=AudioPreproc)
    feat: AudioFeat = field(default_factory=AudioFeat)
    stats: Standardizer = field(default_factory=Standardizer)
    knn: KnnModel = field(default_factory=KnnModel)
    _X_store: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _y_store: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _X_store_raw: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _feature_names: Optional[List[str]] = field(default=None, init=False, repr=False)

    def build_reference_from_paths(self, paths: Sequence[AudioPath], labels: Sequence[str]) -> None:
        """
        ### Construir base de referencia
        Preprocesa rutas, extrae vectores crudos y entrena KNN + Standardizer.
        - Guarda matrices cruda/estandarizada y nombres de features
        ### Resumen
        ```
        orch.build_reference_from_paths(paths, labels)
        ```
        """
        path_list = list(paths)
        label_list = [str(lbl) for lbl in labels]
        if not path_list:
            raise ValueError("paths no puede estar vacío.")
        if len(path_list) != len(label_list):
            raise ValueError("paths y labels deben tener la misma longitud.")

        vectores_raw: list[np.ndarray] = []
        for path in path_list:
            vec_raw, _, _, _ = self._preparar_vector(path, aplicar_estandarizador=False)
            vectores_raw.append(vec_raw.astype(np.float64, copy=False))

        X_raw = np.stack(vectores_raw, axis=0).astype(np.float64, copy=False)
        self._feature_names = self.feat.nombres_de_caracteristicas()
        self.stats.calculate_statistics(X_raw)
        X_std = self.stats.transform(X_raw).astype(np.float32, copy=False)

        self.knn.cargar_referencias(X_std, label_list)
        self._X_store_raw = X_raw
        self._X_store = X_std
        self._y_store = np.asarray(label_list, dtype=np.str_)

    def load_reference_from_repo(self, repo: "Repo", name: str) -> None:
        """
        ### Cargar referencia
        Restaura embeddings y estadísticas desde un repositorio.
        - Actualiza Standardizer y tabla KNN
        ### Resumen
        ```
        orch.load_reference_from_repo(repo, "audio_knn")
        ```
        """
        data = repo.load_knn(name)
        stats = repo.load_model("audio", f"{name}_stats")
        mu = stats["mu"].astype(np.float32, copy=False)
        sigma = stats["sigma"].astype(np.float32, copy=False)
        X = data["X"].astype(np.float32, copy=False)
        y = data["y"].astype(np.str_, copy=False)

        self.stats.mu = mu
        self.stats.sigma = sigma
        self.knn.upload_batch(X, y.tolist())
        self._X_store_raw = None
        self._X_store = X
        self._y_store = y
        self._feature_names = self.feat.nombres_de_caracteristicas()

    def save_reference_to_repo(self, repo: "Repo", name: str) -> None:
        """
        ### Guardar referencia
        Persiste embeddings y estadísticas actuales en el repositorio.
        ### Resumen
        ```
        orch.save_reference_to_repo(repo, "audio_knn")
        ```
        """
        self._ensure_ready()
        repo.save_knn(name, self._X_store, self._y_store)  # type: ignore[arg-type]
        repo.save_model("audio", f"{name}_stats", mu=self.stats.mu, sigma=self.stats.sigma)

    def identify_path(self, path: AudioPath) -> str:
        """
        ### Identificación básica
        Mantiene la interfaz histórica retornando solo la etiqueta.
        ### Resumen
        ```
        label = orch.identify_path("voz.wav")
        ```
        """
        resultado = self.predecir_audio(path, devolver_features=False)
        return resultado["label"]

    def identify_batch(self, paths: Sequence[AudioPath]) -> list[str]:
        """
        ### Identificación por lote
        Devuelve únicamente las etiquetas de cada audio.
        ### Resumen
        ```
        labels = orch.identify_batch(paths)
        ```
        """
        return [res["label"] for res in self.predecir_lote(list(paths))]

    def _ensure_ready(self) -> None:
        """
        ### Chequeo de estado
        Verifica que existan referencias y estadísticas antes de inferir.
        ### Resumen
        ```
        orch._ensure_ready()
        ```
        """
        if self._X_store is None or self._y_store is None or self.stats.mu is None or self.stats.sigma is None:
            raise RuntimeError("Referencia no construida. Llamá a build_reference_from_paths o load_reference_from_repo.")

    def predecir_audio(
        self,
        entrada: Union[AudioPath, Tuple[np.ndarray, int], np.ndarray],
        *,
        devolver_features: bool = True
    ) -> Dict[str, Union[str, float, np.ndarray, List[str], int]]:
        """
        ### Predicción enriquecida
        Devuelve etiqueta, distancia mínima, vectores crudo/estandarizado y audio preprocesado.
        - Controlado por el flag `devolver_features`
        ### Resumen
        ```
        info = orch.predecir_audio("voz.wav", devolver_features=True)
        ```
        """
        self._ensure_ready()
        vec_raw, vec_std, sr, audio_proc = self._preparar_vector(entrada, aplicar_estandarizador=True)
        if vec_std is None:
            raise RuntimeError("Standardizer no está ajustado; no se puede estandarizar el vector.")
        vec_std_f32 = np.asarray(vec_std, dtype=np.float32)
        label = self.knn.predecir(vec_std_f32)
        distancias = self.knn.distancias(vec_std_f32)
        dist_min = float(np.min(distancias))
        nombres = self._feature_names or self.feat.nombres_de_caracteristicas()

        salida: Dict[str, Union[str, float, np.ndarray, List[str], int]] = {
            "label": label,
            "distancia_min": dist_min,
        }
        if devolver_features:
            salida.update({
                "vec_original": vec_raw,
                "vec_estandarizado": np.asarray(vec_std, dtype=np.float64),
                "nombres_de_caracteristicas": nombres,
                "audio_preprocesado": audio_proc,
                "sr": sr,
            })
        return salida

    def predecir_lote(
        self,
        entradas: Sequence[Union[AudioPath, Tuple[np.ndarray, int], np.ndarray]],
        *,
        devolver_features: bool = True
    ) -> List[Dict[str, Union[str, float, np.ndarray, List[str], int]]]:
        """
        ### Predicción por lote
        Itera `predecir_audio` preservando el orden original de la secuencia.
        ### Resumen
        ```
        resultados = orch.predecir_lote(["voz1.wav", "voz2.wav"])
        ```
        """
        items = list(entradas)
        return [self.predecir_audio(item, devolver_features=devolver_features) for item in items]

    def exportar_referencias(self) -> Dict[str, Optional[np.ndarray]]:
        """
        ### Exportar caché
        Retorna matrices cruda/estandarizada y etiquetas almacenadas.
        ### Resumen
        ```
        cache = orch.exportar_referencias()
        ```
        """
        self._ensure_ready()
        return {
            "X_estandarizado": self._X_store,
            "X_crudo": self._X_store_raw,
            "y": self._y_store,
            "nombres_de_caracteristicas": self._feature_names,
        }

    def _preparar_vector(
        self,
        entrada: Union[AudioPath, Tuple[np.ndarray, int], np.ndarray],
        *,
        aplicar_estandarizador: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray], int, np.ndarray]:
        """
        ### Pipeline interno
        Preprocesa la entrada, genera el vector crudo y lo estandariza opcionalmente.
        - Devuelve `(vec_raw, vec_std, sr, audio_proc)`
        ### Resumen
        ```
        vec_raw, vec_std, sr, audio_proc = orch._preparar_vector("voz.wav")
        ```
        """
        if isinstance(entrada, (str, Path)):
            audio_proc, sr = self.preproc.process_path(entrada)
        elif isinstance(entrada, tuple) and len(entrada) == 2:
            y, sr = entrada
            audio_proc, sr = self.preproc.preprocess(np.asarray(y, dtype=np.float32), int(sr))
        elif isinstance(entrada, np.ndarray):
            audio_proc, sr = self.preproc.preprocess(np.asarray(entrada, dtype=np.float32), self.preproc.cfg.target_sr)
        else:
            raise TypeError("Entrada inválida para _preparar_vector.")

        vec_raw = self.feat.extract(audio_proc, sr)
        vec_std: Optional[np.ndarray] = None
        if aplicar_estandarizador and self.stats.mu is not None and self.stats.sigma is not None:
            vec_std = np.asarray(self.stats.transform_one(vec_raw), dtype=np.float64)
        return (
            np.asarray(vec_raw, dtype=np.float64),
            vec_std,
            sr,
            audio_proc,
        )


_DEFAULT_AUDIO_ORCHESTRATOR = AudioOrchestrator()


def build_reference_from_paths(paths: Sequence[AudioPath], labels: Sequence[str]) -> None:
    """Fit the shared audio orchestrator from disk paths."""
    _DEFAULT_AUDIO_ORCHESTRATOR.build_reference_from_paths(paths, labels)


def load_reference_from_repo(repo: "Repo", name: str) -> None:
    """Load audio references into the shared orchestrator from a repository."""
    _DEFAULT_AUDIO_ORCHESTRATOR.load_reference_from_repo(repo, name)


def save_reference_to_repo(repo: "Repo", name: str) -> None:
    """Persist the shared audio reference in the repository."""
    _DEFAULT_AUDIO_ORCHESTRATOR.save_reference_to_repo(repo, name)


def identify_path(path: AudioPath) -> str:
    """Identify the command label for a WAV file stored on disk."""
    return _DEFAULT_AUDIO_ORCHESTRATOR.identify_path(path)


def identify_batch(paths: Sequence[AudioPath]) -> list[str]:
    """Identify a batch of audio file paths."""
    return _DEFAULT_AUDIO_ORCHESTRATOR.identify_batch(paths)

def predecir_audio(entrada: Union[AudioPath, Tuple[np.ndarray, int], np.ndarray], *, devolver_features: bool = True) -> Dict[str, Union[str, float, np.ndarray, List[str], int]]:
    """Wrapper de conveniencia para predecir un solo audio."""
    return _DEFAULT_AUDIO_ORCHESTRATOR.predecir_audio(entrada, devolver_features=devolver_features)

def predecir_lote(entradas: Sequence[Union[AudioPath, Tuple[np.ndarray, int], np.ndarray]], *, devolver_features: bool = True) -> List[Dict[str, Union[str, float, np.ndarray, List[str], int]]]:
    """Wrapper de conveniencia para predecir múltiples audios."""
    return _DEFAULT_AUDIO_ORCHESTRATOR.predecir_lote(entradas, devolver_features=devolver_features)

def exportar_referencias() -> Dict[str, Optional[np.ndarray]]:
    """Obtiene las referencias y metadatos del orquestador global."""
    return _DEFAULT_AUDIO_ORCHESTRATOR.exportar_referencias()


__all__ = [
    "AudioOrchestrator",
    "build_reference_from_paths",
    "load_reference_from_repo",
    "save_reference_to_repo",
    "identify_path",
    "identify_batch",
    "predecir_audio",
    "predecir_lote",
    "exportar_referencias",
]
