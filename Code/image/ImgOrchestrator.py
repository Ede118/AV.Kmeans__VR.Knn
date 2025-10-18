from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, Iterator, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from Code.types import ColorImageU8, GrayImageF32, MatF, MaskU8, VecF
from Code.image.ImgPreproc import ImgPreproc
from Code.image.ImgFeat import ImgFeat
from Code.image.KmeansModel import KMeans

if TYPE_CHECKING:
    from Code.adapters.Repositorio import Repo

ImagePath = Union[str, Path]


@dataclass(slots=True)
class OrchestratorCfg:
    """Configuration knobs for the image pipeline."""

    dim: str = "5D"
    usar_gradiente_en_3D: bool = True
    return_intermediate: bool = False


@dataclass(slots=True)
class ImgOrchestrator:
    """High-level coordinator for the vision pipeline."""

    pre: ImgPreproc = field(default_factory=ImgPreproc)
    feat: ImgFeat = field(default_factory=ImgFeat)
    model: KMeans = field(default_factory=lambda: KMeans(k=4, random_state=42))
    class_names: list[str] = field(default_factory=list)
    class_colors: Dict[str, Tuple[int, int, int]] = field(default_factory=dict)
    oni_tau_global: float = 0.15
    cfg: OrchestratorCfg = field(default_factory=OrchestratorCfg)
    oni_tau_por_cluster: Optional[Dict[int, float]] = None
    cluster_to_label: Dict[int, str] = field(default_factory=dict, init=False)
    _last_fit_features: Optional[MatF] = field(default=None, init=False, repr=False)

    # --- fitting -----------------------------------------------------------------
    def entrenar_desde_rutas(
        self,
        rutas: Sequence[ImagePath],
        *,
        k: int = 4,
        seed: int = 42,
        semillas: Optional[MatF] = None,
    ) -> "ImgOrchestrator":
        """
        ### Entrenamiento desde rutas
        Procesa cada imagen en un vector float64, entrena K-Means y calcula τ ONI por cluster.
        - Normaliza/segmenta cada ruta mediante `_preparar_vector`
        - Reajusta la etiqueta mayoritaria y el mapeo cluster→nombre
        - Si hay ≥5 muestras por cluster, usa percentil 95 de distancia; de lo contrario, `oni_tau_global`
        ### Resumen

        ```
        orch = ImgOrchestrator()
        orch.entrenar_desde_rutas(lista_de_rutas, k=4, seed=42)
        ```
        """

        dataset = list(rutas)
        if not dataset:
            raise ValueError("paths no puede estar vacío.")
        caracteristicas, etiquetas = self._preparar_caracteristicas(dataset)
        if caracteristicas.shape[0] < k:
            raise ValueError("k no puede exceder la cantidad de imágenes disponibles.")
        self.model = KMeans(k=k, random_state=seed)
        semillas_fit = np.asarray(semillas, dtype=np.float64) if semillas is not None else None
        self.model.ajustar(caracteristicas, semillas=semillas_fit)
        asignaciones, distancias = self.model.predecir_con_distancias(caracteristicas)
        self.cluster_to_label = self._build_mapping(asignaciones, etiquetas)
        self.class_names = [self.cluster_to_label.get(i, f"cluster_{i}") for i in range(self.model.k)]
        self.oni_tau_por_cluster = self._calibrar_oni(asignaciones, distancias)
        self._last_fit_features = caracteristicas
        return self

    def fit_from_paths(
        self,
        paths: Sequence[ImagePath],
        *,
        k: int = 4,
        seed: int = 42,
        seeds: Optional[MatF] = None,
    ) -> "ImgOrchestrator":
        """
        ### Alias histórico
        Delegación hacia `entrenar_desde_rutas` con los mismos parámetros.
        - Mantiene compatibilidad con llamadas previas
        - Reutiliza toda la lógica de entrenamiento/calibración
        ### Resumen

        ```
        orch.fit_from_paths(paths, k=4, seed=42)
        ```
        """
        
        return self.entrenar_desde_rutas(paths, k=k, seed=seed, semillas=seeds)

    def train(
            self, 
            matParametros: MatF, 
            seeds: Optional[MatF] = None
            ) -> "ImgOrchestrator":
        """
        ### Entrenamiento con features precalculados
        Acepta matrices N×D ya generadas y entrena sin releer imágenes.
        - Requiere `matParametros` en float64
        - Actualiza τ ONI por cluster usando los vectores entregados
        ### Resumen

        ```
        orch.train(features_float64)
        ```
        """
        X = np.asarray(matParametros, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("matParametros debe tener shape (N, D)")
        fit_seeds = np.asarray(seeds, dtype=np.float64) if seeds is not None else None
        self.model.ajustar(X, semillas=fit_seeds)
        self.cluster_to_label = {i: f"cluster_{i}" for i in range(self.model.k)}
        self.class_names = [self.cluster_to_label[i] for i in range(self.model.k)]
        etiquetas, distancias = self.model.predecir_con_distancias(X)
        self.oni_tau_por_cluster = self._calibrar_oni(etiquetas, distancias)
        self._last_fit_features = X
        return self

    def train_from_path(
        self,
        paths: Sequence[ImagePath],
        seeds: Optional[MatF] = None,
    ) -> "ImgOrchestrator":
        """
        ### Alias retrocompatible
        Reutiliza `entrenar_desde_rutas` manteniendo la firma original.
        - Usa `self.model.k` y `self.model.random_state` como parámetros por defecto
        ### Resumen

        ```
        orch.train_from_path(paths, seeds=None)
        ```
        """
        return self.entrenar_desde_rutas(paths, k=self.model.k, seed=self.model.random_state or 42, semillas=seeds)

    def train_from_patron(
        self,
        pattern: str,
        start: int,
        end: int,
        seeds: Optional[MatF] = None,
    ) -> "ImgOrchestrator":
        """
        ### Expansión de patrón
        Genera rutas con formato printf y entrena el modelo.
        - Valida rangos y extensiones soportadas
        - Llama internamente a `entrenar_desde_rutas`
        ### Resumen

        ```
        orch.train_from_patron("imgs/pieza_%03d.png", 0, 50)
        ```
        """
        if "%d" not in pattern and "%i" not in pattern:
            raise ValueError("El patrón debe contener %d o %i (o con padding).")
        if start > end:
            raise ValueError("start debe ser <= end")
        paths = [pattern % i for i in range(start, end + 1)]
        valid = [p for p in paths if Path(p).suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}]
        if not valid:
            raise RuntimeError("El patrón no generó rutas válidas.")
        return self.entrenar_desde_rutas(valid, k=self.model.k, seed=self.model.random_state or 42, semillas=seeds)

    # --- prediction --------------------------------------------------------------
    def identify(
            self, 
            img: np.ndarray
            ) -> str:
        """
        ### Identificación directa
        Obtiene el nombre de clase para una imagen ya cargada.
        - Delegación a `predecir_imagen`
        - Devuelve el nombre mapeado por `cluster_to_label`
        ### Resumen

        ```
        label = orch.identify(img_ndarray)
        ```
        """
        salida = self.predecir_imagen(img)
        return salida["class_name"]

    def identify_path(
            self, 
            path: ImagePath
            ) -> str:
        """
        ### Identificación desde ruta
        Lee la imagen, infiere el vector y retorna la etiqueta textual.
        - Usa `_cargar` y `predecir_imagen`
        ### Resumen

        ```
        label = orch.identify_path("ruta/pieza.png")
        ```
        """
        salida = self.predecir_imagen(path)
        return salida["class_name"]

    def identify_batch(
            self, 
            paths: Sequence[ImagePath]
            ) -> list[str]:
        """
        ### Identificación en lote
        Procesa una secuencia de rutas y devuelve la lista de nombres.
        - Internamente usa `predecir_lote` para conservar metadatos
        ### Resumen

        ```
        etiquetas = orch.identify_batch(lista_de_rutas)
        ```
        """
        return [resultado["class_name"] for resultado in self.predecir_lote(paths)]

    def load_centroids_from_file(
            self, 
            path: ImagePath
            ) -> "ImgOrchestrator":
        """
        ### Carga de centroides
        Restaura el modelo a partir de un archivo `.npz` con la clave `C`.
        - Reemplaza KMeans y reinicia metadatos asociados
        - Mapea cada cluster a `cluster_{k}` por defecto
        ### Resumen

        ```
        orch.load_centroids_from_file("centroides.npz")
        ```
        """
        npz_path = Path(path)
        if not npz_path.is_file():
            raise FileNotFoundError(f"No se encontró el archivo de centroides: {npz_path}")
        with np.load(npz_path, allow_pickle=False) as data:
            if "C" not in data:
                raise KeyError("El archivo no contiene la clave 'C'.")
            centroids = data["C"].astype(np.float64, copy=False)
        self.model = KMeans(k=int(centroids.shape[0]), random_state=self.model.random_state)
        self.model.matCentroides_ = centroids
        self.model.vecLabels_ = None
        self.model.inercia_ = None
        self.cluster_to_label = {i: f"cluster_{i}" for i in range(self.model.k)}
        self.class_names = [self.cluster_to_label[i] for i in range(self.model.k)]
        self.oni_tau_por_cluster = None
        self._last_fit_features = None
        return self

    def predecir_imagen(
            self, 
            input_: Union[ImagePath, ColorImageU8]
            ) -> Dict[str, Any]:
        """
        ### Predicción completa
        Calcula vector de características y obtiene cluster/distancia.
        - Devuelve payload con `vecF`, `vecF_nombrado`, color y τ ONI
        - Incluye intermedios (`img_norm`, `mask`) si `return_intermediate`
        ### Resumen

        ```
        info = orch.predecir_imagen("ruta.png")
        ```
        """
        vec, img_norm, mask = self._procesar_para_vector(input_)
        etiquetas, distancias = self.model.predecir_con_distancias(vec[np.newaxis, :])
        idx = int(etiquetas[0])
        label = self.cluster_to_label.get(idx, f"cluster_{idx}")
        color = self.class_colors.get(label, (128, 128, 128))
        nombres = self.feat.nombres_de_caracteristicas(dim=self.cfg.dim, usar_gradiente_en_3D=self.cfg.usar_gradiente_en_3D)
        vec_nombrado = {nombre: float(valor) for nombre, valor in zip(nombres, vec.tolist())}
        tau_cluster = None
        if self.oni_tau_por_cluster is not None:
            tau_cluster = self.oni_tau_por_cluster.get(idx, self.oni_tau_global)
        payload: Dict[str, Any] = {
            "class_idx": idx,
            "class_name": label,
            "color": color,
            "d2_min": float(distancias[0]),
            "vecF": vec,
            "vecF_nombrado": vec_nombrado,
            "meta": {
                "dim": self.cfg.dim,
                "usar_gradiente_en_3D": self.cfg.usar_gradiente_en_3D,
                "preproc": self.pre.cfg,
            },
        }
        if tau_cluster is not None:
            payload["oni_tau"] = float(tau_cluster)
        if self.cfg.return_intermediate:
            payload["intermediate"] = {"img_norm": img_norm, "mask": mask}
        return payload

    def predecir_lote(
            self, 
            inputs: Union[int, Sequence[Union[ImagePath, ColorImageU8]]]
            ) -> list[Dict[str, Any]]:
        """
        ### Predicción por lote
        Acepta entero (muestras recientes) o lista de entradas heterogéneas.
        - Para `int`, reutiliza `_last_fit_features`
        - Para secuencias, aplica `predecir_imagen` sobre cada elemento
        ### Resumen

        ```
        resultados = orch.predecir_lote(["a.png", "b.png"])
        muestras = orch.predecir_lote(5)
        ```
        """
        if isinstance(inputs, int):
            if self._last_fit_features is None:
                raise RuntimeError("No hay features recientes. Llamá a fit_from_paths primero.")
            if inputs <= 0:
                raise ValueError("n debe ser > 0")
            total = min(inputs, self._last_fit_features.shape[0])
            indices = np.arange(total)
            vectores = np.asarray(self._last_fit_features[indices], dtype=np.float64)
            etiquetas, distancias = self.model.predecir_con_distancias(vectores)
            nombres = self.feat.nombres_de_caracteristicas(dim=self.cfg.dim, usar_gradiente_en_3D=self.cfg.usar_gradiente_en_3D)
            resultados: list[Dict[str, Any]] = []
            for vec, etiqueta, distancia in zip(vectores, etiquetas, distancias):
                idx = int(etiqueta)
                label = self.cluster_to_label.get(idx, f"cluster_{idx}")
                color = self.class_colors.get(label, (128, 128, 128))
                vec = np.asarray(vec, dtype=np.float64).reshape(-1)
                vec_nombrado = {nombre: float(valor) for nombre, valor in zip(nombres, vec.tolist())}
                tau_cluster = None
                if self.oni_tau_por_cluster is not None:
                    tau_cluster = self.oni_tau_por_cluster.get(idx, self.oni_tau_global)
                payload: Dict[str, Any] = {
                    "class_idx": idx,
                    "class_name": label,
                    "color": color,
                    "d2_min": float(distancia),
                    "vecF": vec,
                    "vecF_nombrado": vec_nombrado,
                    "meta": {
                        "dim": self.cfg.dim,
                        "usar_gradiente_en_3D": self.cfg.usar_gradiente_en_3D,
                        "preproc": self.pre.cfg,
                    },
                }
                if tau_cluster is not None:
                    payload["oni_tau"] = float(tau_cluster)
                resultados.append(payload)
            return resultados
        if isinstance(inputs, (str, Path, np.ndarray)):
            raise TypeError("predecir_lote espera una secuencia de rutas o imágenes.")
        items = list(inputs)
        resultados = [self.predecir_imagen(item) for item in items]
        return resultados

    def run_from_dir(
        self,
        root: ImagePath,
        recursive: bool = True,
        exts: Iterable[str] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"),
    ) -> list[Dict[str, Any]]:
        """
        ### Predicción desde directorio
        Escanea el árbol de archivos y ejecuta la inferencia para cada imagen válida.
        - Filtra extensiones y respeta el flag `recursive`
        - Devuelve la salida de `predecir_lote`
        ### Resumen

        ```
        reporte = orch.run_from_dir("data/test", recursive=False)
        ```
        """
        base = Path(root)
        if not base.exists():
            raise FileNotFoundError(f"No existe el directorio: {base}")
        it: Iterator[Path] = base.rglob("*") if recursive else base.glob("*")
        valid = sorted(p for p in it if p.suffix.lower() in {e.lower() for e in exts})
        if not valid:
            raise RuntimeError(f"No encontré imágenes en: {base}")
        return self.run_batch(valid)

    # --- helpers -----------------------------------------------------------------
    run = predecir_imagen
    run_batch = predecir_lote
    predict_batch = predecir_lote

    def _cargar(
            self,
            entrada: Union[ImagePath, ColorImageU8]
            ) -> ColorImageU8:
        """
        ### Loader interno
        Convierte rutas o arreglos en imágenes BGR uint8.
        - Usa `cv2.imread` para strings/Path
        - Valida `ndarray` (H×W×3, dtype=uint8)
        """
        if isinstance(entrada, (str, Path)):
            img = cv2.imread(str(entrada), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"No pude leer la imagen: {entrada}")
            return img
        arr = np.asarray(entrada)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise TypeError("La imagen debe tener formato HxWx3.")
        if arr.dtype != np.uint8:
            raise TypeError("La imagen debe ser uint8 en el rango [0,255].")
        return arr

    _load = _cargar

    def _read_image(
            self,
            path: ImagePath
            ) -> ColorImageU8:
        return self._cargar(path)

    def _procesar_para_vector(
            self,
            entrada: Union[ImagePath, ColorImageU8]
            ) -> Tuple[np.ndarray, GrayImageF32, MaskU8]:
        """
        ### Pipeline interno
        Normaliza, segmenta y extrae el vector de características.
        - Retorna `(vec_float64, img_norm, mask)` listos para predicción
        """
        img_bgr = self._cargar(entrada)
        img_norm = self.pre.normalize(img_bgr)
        mask = self.pre.segment(img_norm).astype(np.uint8, copy=False)
        vec = self.feat.shape_vector(
            img_norm,
            mask,
            dim=self.cfg.dim,
            usar_gradiente_en_3D=self.cfg.usar_gradiente_en_3D,
        )
        vec = np.asarray(vec, dtype=np.float64).reshape(-1)
        return vec, img_norm, mask

    def _preparar_vector(
            self,
            entrada: Union[ImagePath, ColorImageU8]
            ) -> VecF:
        """
        ### Vector único
        Devuelve solo el descriptor float64 asociado a la entrada.
        - Alias sobre `_procesar_para_vector`
        """
        vec, _, _ = self._procesar_para_vector(entrada)
        return vec.copy()

    def _extract_feature_from_array(
            self, 
            img: np.ndarray
            ) -> VecF:
        """
        ### Compatibilidad histórica
        Mantiene la firma previa para obtener el vector desde `np.ndarray`.
        - Devuelve copia float64 en forma 1D
        """
        vec, _, _ = self._procesar_para_vector(img)
        return vec.copy()

    def _preparar_caracteristicas(
            self, 
            paths: Sequence[ImagePath]
            ) -> Tuple[MatF, list[str]]:
        """
        ### Construcción de dataset
        Convierte rutas en matriz N×D float64 y etiqueta por carpeta.
        - Reutiliza `_preparar_vector` para cada elemento
        """
        
        features: list[np.ndarray] = []
        labels: list[str] = []
        
        for path in paths:
            vec = self._preparar_vector(path)
            features.append(vec.reshape(-1))
            label = Path(path).parent.name.strip().lower()
            labels.append(label if label else f"cluster_{len(labels)}")
        
        X = np.vstack(features).astype(np.float64, copy=False)
        
        return X, labels

    _prepare_dataset = _preparar_caracteristicas

    def _calibrar_oni(
            self, 
            etiquetas: np.ndarray, 
            distancias: np.ndarray
            ) -> Dict[int, float]:
        """
        ### Calibración ONI
        Calcula τ cluster-wise (percentil 95) con fallback global.
        - Necesita etiquetas y distancias cuadráticas como arrays
        """
        etiquetas = np.asarray(etiquetas, dtype=np.int64)
        distancias = np.asarray(distancias, dtype=np.float64)
        resultado: Dict[int, float] = {}
        for cluster in range(self.model.k):
            mascara = etiquetas == cluster
            if np.count_nonzero(mascara) >= 5:
                tau = float(np.percentile(distancias[mascara], 95))
            else:
                tau = float(self.oni_tau_global)
            resultado[cluster] = tau
        return resultado

    def _build_mapping(
            self, 
            assignments: np.ndarray, 
            labels: Sequence[str]
            ) -> Dict[int, str]:
        """
        ### Etiquetado mayoritario
        Asigna nombre por cluster en base a frecuencia de etiquetas.
        - Fallback: `cluster_{k}`
        """
        mapping: Dict[int, str] = {}
        for cluster in range(self.model.k):
            assigned = [labels[i] for i, c in enumerate(assignments) if int(c) == cluster]
            if assigned:
                major = Counter(assigned).most_common(1)[0][0]
                mapping[cluster] = major
            else:
                mapping[cluster] = f"cluster_{cluster}"
        return mapping


# ----------------------------- module level convenience -----------------------------
_DEFAULT_ORCHESTRATOR = ImgOrchestrator()


def fit_from_paths(paths: Sequence[ImagePath], *, k: int = 4, seed: int = 42) -> None:
    """Fit the shared orchestrator from raw image paths."""
    _DEFAULT_ORCHESTRATOR.fit_from_paths(paths, k=k, seed=seed)


def load_centroids_from_file(path: ImagePath) -> None:
    """Load centroids into the shared orchestrator."""
    _DEFAULT_ORCHESTRATOR.load_centroids_from_file(path)


def identify(img: np.ndarray) -> str:
    """Identify the class name for an already-loaded image."""
    return _DEFAULT_ORCHESTRATOR.identify(img)


def identify_path(path: ImagePath) -> str:
    """Identify the class for an image stored on disk."""
    return _DEFAULT_ORCHESTRATOR.identify_path(path)


def identify_batch(paths: Sequence[ImagePath]) -> list[str]:
    """Identify a batch of image paths."""
    return _DEFAULT_ORCHESTRATOR.identify_batch(paths)


def fit_from_repo(repo: "Repo", split: str = "train") -> None:
    """Fit the shared orchestrator using the paths yielded by a repository split."""
    paths = list(repo.iter_split("vision", split))  # type: ignore[attr-defined]
    fit_from_paths(paths, k=_DEFAULT_ORCHESTRATOR.model.k, seed=_DEFAULT_ORCHESTRATOR.model.random_state or 42)


def identify_from_repo(repo: "Repo", split: str = "test", n: Optional[int] = None) -> list[str]:
    """Identify images coming from a repository split."""
    paths = list(repo.iter_split("vision", split))  # type: ignore[attr-defined]
    if n is not None:
        paths = paths[:n]
    return identify_batch(paths)


__all__ = [
    "ImgOrchestrator",
    "OrchestratorCfg",
    "fit_from_paths",
    "load_centroids_from_file",
    "identify",
    "identify_path",
    "identify_batch",
    "fit_from_repo",
    "identify_from_repo",
]
