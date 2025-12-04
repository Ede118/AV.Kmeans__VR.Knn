from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from Code.types import ImgColor, ImgGray, Mask, MatF, VecF
from Code.image.ImgPreproc import ImgPreproc, ImgPreprocCfg
from Code.image.ImgFeat import ImgFeat, hyper_params
from Code.image.KmeansModel import KMeansModel

ImagePath = str | Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


@dataclass(slots=True)
class OrchestratorCfg:
	"""Configuration knobs for the image pipeline."""

	dim: str = "3D"
	usar_gradiente_en_3D: bool = False
	return_intermediate: bool = False


@dataclass(slots=True)
class ImgOrchestrator:
	"""High-level coordinator for the vision pipeline."""

	pre: ImgPreproc = field(
		default_factory=lambda: ImgPreproc(
			cfg=ImgPreprocCfg(
				target_size=512,
				sigma=2.0,
				flag_refine_mask=True,
				open_ksize=5,
				close_ksize=5,
			)
		)
	)

	feat: ImgFeat = field(
		default_factory=lambda: ImgFeat(
			hp=hyper_params(
				radial_var_t_low=0.0,
				radial_var_t_high=0.045,
				r_hull_t=0.225,
			),
			mode="3D",
		)
	)

	model: KMeansModel = field(
		default_factory=lambda: KMeansModel(n_clusters=4, init_centers=None, random_state=42)
	)

	cfg: OrchestratorCfg = field(default_factory=OrchestratorCfg)
	class_names: list[str] = field(default_factory=list)
	class_colors: dict[str, tuple[int, int, int]] = field(default_factory=dict)
	feature_names: list[str] = field(default_factory=list, init=False)
	oni_tau_global: float = 0.15
	oni_tau_por_cluster: dict[int, float] | None = None
	cluster_to_label: dict[int, str] = field(default_factory=dict, init=False)
	_last_fit_features: MatF | None = field(default=None, init=False, repr=False)

	# ------------------------------------------------------------------ #
	# Entrenamiento, carga y guardado de modelos
	# ------------------------------------------------------------------ #
	def entrenar(
		self,
		dataset_dir: ImagePath,
		*,
		run: int = 1,
		seeds: np.ndarray | None = None,
		output_root: ImagePath | None = None,
		labels: Sequence[str] | None = None,
	) -> dict[str, Any]:
		"""
		Pipeline completo estilo notebook:
		- Lee imágenes de subcarpetas dentro de `dataset_dir`.
		- Preprocesa y guarda recortes/máscaras.
		- Extrae features, exporta CSV (uno por alpha) y entrena K-Means.

		Parameters
		----------
		dataset_dir : str | Path
			Carpeta con subcarpetas por clase (p.ej. `Database/data/image7`).
		run : int
			Número usado para nombrar la carpeta de salida (`Model Intento XX`).
		seeds : np.ndarray | None
			Centros iniciales opcionales (shape (k, F)). Si se proveen, definen k.
		output_root : str | Path | None
			Carpeta base para salidas. Default: `PROJECT_ROOT/Database/tmp/image`.
		labels : sequence[str] | None
			Si se indica, solo usa las carpetas con esos nombres (case-insensitive).
		"""
		dataset_path = Path(dataset_dir)
		if not dataset_path.is_absolute():
			candidate = (PROJECT_ROOT / dataset_path).resolve()
			if candidate.exists():
				dataset_path = candidate
		if not dataset_path.exists():
			raise FileNotFoundError(f"No existe el dataset: {dataset_path}")

		label_set = None
		if labels is not None:
			label_set = {str(l).strip().lower() for l in labels if str(l).strip()}

		paths = sorted(
			p for p in dataset_path.glob("*/*")
			if p.is_file()
			and p.suffix.lower() in {e.lower() for e in VALID_EXTS}
			and (label_set is None or p.parent.name.lower() in label_set)
		)
		if not paths:
			raise RuntimeError(f"No encontré imágenes dentro de {dataset_path}")

		base_out = Path(output_root) if output_root is not None else PROJECT_ROOT / "Database" / "tmp" / "image"
		if not base_out.is_absolute():
			base_out = (PROJECT_ROOT / base_out).resolve()
		out_path = base_out if output_root is not None else base_out / f"Model Intento {run:02d}"
		out_path.mkdir(parents=True, exist_ok=True)

		rows: list[list[Any]] = []
		features: list[np.ndarray] = []
		etiquetas_str: list[str] = []
		names: Optional[list[str]] = None

		for p in paths:
			img_bgr = cv2.imread(str(p))
			if img_bgr is None:
				continue

			img_sq, mask_sq = self.pre.procesar(img_bgr, blacknwhite=False)

			clase_dir = out_path / p.parent.name
			clase_dir.mkdir(parents=True, exist_ok=True)
			cv2.imwrite(str(clase_dir / f"{p.stem} recortada.png"), img_sq)
			cv2.imwrite(str(clase_dir / f"{p.stem} mask.png"), mask_sq)

			vec, names_local, _ = self.feat.extraer_features(img_sq, mask_sq)
			vec = np.asarray(vec, dtype=np.float64).reshape(-1)
			if names is None:
				names = list(names_local)
				self.feature_names = names
			rows.append([p.parent.name, p.name, *vec.tolist()])
			features.append(vec)
			etiquetas_str.append(p.parent.name)

		if names is None or not features:
			raise RuntimeError("No se pudieron extraer features del dataset.")

		X = np.vstack(features).astype(np.float64, copy=False)
		df_meta = pd.DataFrame(rows, columns=["clase", "archivo", *names])[["clase", "archivo"]]
		df_base = pd.DataFrame(X, columns=names)
		df_out = pd.concat([df_meta.reset_index(drop=True), df_base], axis=1)
		csv_paths: list[Path] = []
		csv_path = out_path / "features.csv"
		df_out.to_csv(csv_path, index=False)
		csv_paths.append(csv_path)


		k = seeds.shape[0] if seeds is not None else max(len(set(etiquetas_str)), self.model.n_clusters)
		k = min(k, X.shape[0])
		seeds_fit = np.asarray(seeds, dtype=np.float64) if seeds is not None else None
		self.model = KMeansModel(n_clusters=k, random_state=self.model.random_state, init_centers=None)
		self.model.fit(X, init_centers=seeds_fit)
		asignaciones, distancias = self.model.predecir_con_distancias(X)

		self.cluster_to_label = self._build_mapping(asignaciones, etiquetas_str)
		self.class_names = [self.cluster_to_label.get(i, f"cluster_{i}") for i in range(self.model.n_clusters)]
		self.oni_tau_por_cluster = self._calibrar_oni(asignaciones, distancias)
		self._last_fit_features = X

		seed_tag = "manual" if seeds is not None else "kmeans++"
		df_runs = pd.DataFrame([{"seed": seed_tag, "inertia": self.model._inertia}])
		centers = self.model._centers if self.model._centers is not None else np.zeros((k, X.shape[1]), dtype=np.float64)
		rows_centers: list[dict[str, Any]] = []
		for idx, center in enumerate(centers):
			rows_centers.append({"seed": seed_tag, "cluster": idx, **{n: center[i] for i, n in enumerate(names)}})
		df_centers = pd.DataFrame(rows_centers)

		return {
			"out_dir": out_path,
			"csv_paths": csv_paths,
			"df_features": df_out,
			"df_runs": df_runs,
			"df_centers": df_centers,
			"labels": asignaciones,
			"centers": centers,
		}

	# --- prediction --------------------------------------------------------------
	def guardar_modelo(
		self,
		path: ImagePath
		) -> "ImgOrchestrator":
		"""
		### Guardado de centroides
		Persiste los centroides del K-Means en un `.npz` con la clave `C`.
		- Crea la carpeta destino si no existe.
		- Requiere que el modelo ya esté entrenado (`_centers` definido).
		### Resumen

		```
		orch.guardar_modelo("centroides.npz")
		```
		"""
		if self.model._centers is None:
			raise RuntimeError("No hay centroides para guardar; entrená el modelo primero.")
		npz_path = Path(path)
		npz_path.parent.mkdir(parents=True, exist_ok=True)
		np.savez(npz_path, C=self.model._centers.astype(np.float32, copy=False))
		return self

	def cargar_modelo(
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
		
		self.model = KMeansModel(n_clusters=int(centroids.shape[0]), random_state=self.model.random_state)
		self.model.centers_ = centroids
		self.model.inertia_ = None
		self.cluster_to_label = {i: f"cluster_{i}" for i in range(self.model.n_clusters)}
		self.class_names = [self.cluster_to_label[i] for i in range(self.model.n_clusters)]
		self.oni_tau_por_cluster = None
		self._last_fit_features = None
		
		return self

	def predecir(
		self,
		input_dir: ImagePath,
		*,
		recursive: bool = True,
		exts: Iterable[str] = VALID_EXTS,
		return_df: bool = True,
	) -> Union[list[dict[str, Any]], tuple[list[dict[str, Any]], pd.DataFrame]]:
		"""
		Predice clases para todas las imágenes en un directorio.

		- Recorre recursivamente `input_dir` y toma extensiones válidas.
		- Devuelve una lista de dicts con `cluster_number` y `label_name` (y el path).
		- Si `return_df=True`, también devuelve un DataFrame compactado.
		"""
		if self.model._centers is None:
			raise RuntimeError("Entrena el modelo antes de predecir (llamá a entrenar_desde_directorio).")

		base = Path(input_dir)
		if not base.is_absolute():
			candidate = (PROJECT_ROOT / base).resolve()
			if candidate.exists():
				base = candidate
		if not base.exists():
			raise FileNotFoundError(f"No existe el directorio: {base}")

		it: Iterator[Path] = base.rglob("*") if recursive else base.glob("*")
		valid_exts = {e.lower() for e in exts}
		paths = sorted(p for p in it if p.suffix.lower() in valid_exts)
		if not paths:
			raise RuntimeError(f"No encontré imágenes en: {base}")

		vectores: list[np.ndarray] = []
		for p in paths:
			img = cv2.imread(str(p))
			if img is None:
				continue
			vec, _, _ = self._procesar_para_vector(img)
			vectores.append(vec)

		if not vectores:
			raise RuntimeError("No se pudo procesar ninguna imagen en el directorio.")

		X = np.vstack(vectores).astype(np.float64, copy=False)
		etiquetas = self.model.predict(X)
		resultado: list[dict[str, Any]] = []
		for i, (ruta, idx) in enumerate(zip(paths, etiquetas), start=1):
			label = self.cluster_to_label.get(int(idx), f"cluster_{int(idx)}")
			path_rel = str(ruta.relative_to(base)) if ruta.is_relative_to(base) else str(ruta)
			resultado.append(
				{
					"id": i,
					"path relativo": path_rel,
					"Número de Cluster": int(idx),
					"Clasificación": label,
				}
			)
		if return_df:
			df = pd.DataFrame(resultado, columns=["id", "path relativo", "Número de Cluster", "Clasificación"])
			return resultado, df
		return resultado

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

	def _procesar_para_vector(
			self,
			entrada: Union[ImagePath, ColorImageU8]
			) -> tuple[np.ndarray, GrayImageF32, MaskU8]:
		"""
		### Pipeline interno
		Normaliza, segmenta y extrae el vector de características.
		- Retorna `(vec_float64, img_norm, mask)` listos para predicción
		"""
		img_bgr = self._cargar(entrada)
		img_norm, mask = self.pre.procesar(img_bgr)
		mask = np.asarray(mask, dtype=np.uint8, copy=False)
		vec, names, debug = self.feat.extraer_features(img_norm=img_norm, mask=mask)
		self.feature_names = list(names)
		vec = np.asarray(vec, dtype=np.float64).reshape(-1)
		return vec, img_norm, mask

	def _preparar_caracteristicas(
			self, 
			paths: sequence[ImagePath]
			) -> tuple[MatF, list[str]]:
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
			label = Path(path).parent.name.strip()
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
		for cluster in range(self.model.n_clusters):
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
			labels: sequence[str]
			) -> Dict[int, str]:
		"""
		### Etiquetado mayoritario
		Asigna nombre por cluster en base a frecuencia de etiquetas.
		- Fallback: `cluster_{k}`
		"""
		mapping: Dict[int, str] = {}
		for cluster in range(self.model.n_clusters):
			assigned = [labels[i] for i, c in enumerate(assignments) if int(c) == cluster]
			if assigned:
				major = Counter(assigned).most_common(1)[0][0]
				mapping[cluster] = major
			else:
				mapping[cluster] = f"cluster_{cluster}"
		return mapping


# ----------------------------- module level convenience -----------------------------
_DEFAULT_ORCHESTRATOR = ImgOrchestrator()


def load_centroids_from_file(path: ImagePath) -> None:
	"""Load centroids into the shared orchestrator."""
	_DEFAULT_ORCHESTRATOR.cargar_modelo(path)


__all__ = [
	"ImgOrchestrator",
	"OrchestratorCfg",
	"load_centroids_from_file",
]
