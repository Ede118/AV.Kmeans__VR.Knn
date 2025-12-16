from __future__ import annotations
from dataclasses import dataclass, field

import cv2
import numpy as np

@dataclass(slots=True)
class hyper_params:
	"""Umbrales heurísticos para normalizar variaciones geométricas."""
	radial_var_t_low: float = 0.05
	radial_var_t_high: float = 0.0
	r_hull_t: float = 0.3


@dataclass(slots=True)
class ImgFeat:
	"""Extracción de features geométricos desde imagen y máscara."""
	hp: hyper_params = field(default_factory=hyper_params)
	mode: str = "3D"
	use_meta: bool = False

	def __post_init__(self) -> None:
		"""Normaliza `mode` y valida que sea uno de los modos soportados (3D o 6D)."""
		self.mode = self.mode.upper()
		if self.mode not in {"3D", "6D"}:
			raise ValueError(...)

	# ------------------------------------------------------------------ #
	@staticmethod
	def feature_names(mode: str = "3D") -> list[str]:
		"""Devuelve los nombres de features en el orden exacto del vector."""

		mode_norm = mode.upper()
		if mode_norm == "3D":
			return [
				"huecos",
				"r hull",
				"variacion radial",
			]
		if mode_norm == "6D":
			return [
				"huecos",
				"r hull",
				"variacion radial",
				"circularidad",
				"solidez",
				"ratio de inercia",
			]
		raise ValueError(f"mode debe ser '3D' o '6D', no '{mode}'")

	# Compat alias ---------------------------------------------------------------- #
	def nombres_de_caracteristicas(
		self,
		dim: str = "3D",
		usar_gradiente_en_3D: bool = False,
	) -> list[str]:
		"""
		Alias en español para recuperar los nombres de features.

		Parameters
		----------
		dim : str
			Dimensión solicitada. Usa el valor del orquestador; se ignora
			`usar_gradiente_en_3D` porque el modo actual solo diferencia 3D/6D.
		usar_gradiente_en_3D : bool
			Conservado por compatibilidad; no altera la salida.
		"""
		_ = usar_gradiente_en_3D  # compatibilidad, sin efecto
		return self.feature_names(dim)

	# ------------------------------------------------------------------ #
	def extraer_features(
		self,
		img_norm: ImgColor,
		mask: Mask
	) -> tuple[VecF, list[str], dict[str, object]]:
		"""
		Calcula el vector de features y metadatos de apoyo.

		Parameters
		----------
		img_norm : np.ndarray
			Imagen float32 ∈ [0,1] alineada y centrada (salida de ImgPreproc).
		mask : np.ndarray
			Máscara uint8 {0,255} (o convertible) alineada con `img_norm`.

		Returns
		-------
		vec : np.ndarray
			Vector float32 de dimensión 5 o 7 dependiendo del modo.
		names : list[str]
			Nombres de cada componente del vector, en el mismo orden.
		debug : dict
			Medidas intermedias útiles para depuración.
		"""

		names = self.feature_names(self.mode)
		n_dim = len(names)
		mask = self._2mask(mask)
		debug: dict[str, object] = {}

		if mask is None or cv2.countNonZero(mask) == 0:
			debug["empty_mask"] = True
			return np.zeros(n_dim, dtype=np.float32), names, debug

		img_F32 = np.asarray(img_norm, dtype=np.float32)

		contour = None
		rect = None
		inertia_ratio = None
		holes_meta = None
		contour_from_mask = False
		hole_indices: list[int] = []

		if contour is None:
			contour, hierarchy, hole_indices = self._contorno_jerarquia(mask)
			contour_from_mask = True
			if contour is None or len(contour) < 3:
				debug["empty_contour"] = True
				return np.zeros(n_dim, dtype=np.float32), names, debug
		else:
			hierarchy = None  # reutilizaremos la máscara para métricas de huecos

		area, perimeter = self._contorno_area_y_perimetro(contour)
		mask_area = float(cv2.countNonZero(mask))
		hull, hull_perimeter = self._hull_convexo_y_perimetro(contour)
		hull_area = float(cv2.contourArea(hull)) if hull is not None else 0.0

		if rect is None:
			rect = cv2.minAreaRect(contour)

		centroid = tuple(rect[0]) if rect else (0.0, 0.0)
		radial_var = self._variacion_radial(contour, centroid)

		if inertia_ratio is None:
			inertia_ratio = self._inercia_desde_momentos(contour)

		if holes_meta is not None:
			huecos = int(holes_meta)
		elif hierarchy is not None:
			huecos = len(hole_indices)
		else:
			contour_tmp, hierarchy_tmp, hole_indices = self._contorno_jerarquia(mask)
			huecos = len(hole_indices)
			if contour is None and contour_tmp is not None:
				contour = contour_tmp
				area, perimeter = self._contorno_area_y_perimetro(contour)
				hull, hull_perimeter = self._hull_convexo_y_perimetro(contour)
				hull_area = float(cv2.contourArea(hull)) if hull is not None else 0.0
				rect = cv2.minAreaRect(contour)

		if huecos > 1:
			huecos = 1
		

		hole_area = max(area - mask_area, 0.0)
		hole_area_ratio = hole_area / (area + 1e-9)
		circularity = (4.0 * np.pi * area) / (perimeter * perimeter + 1e-9)
		solidity = area / (hull_area + 1e-9) if hull_area > 0 else 0.0
		
		r_hull = ((perimeter / (hull_perimeter + 1e-9)) - 1.0)
		r_hull = min(max(r_hull, 0.0), self.hp.r_hull_t)
		r_hull = r_hull / self.hp.r_hull_t

		inner_gradient = self._gradiente_interno(img_F32, mask)

		debug.update(
			area=area,
			perimeter=perimeter,
			mask_area=mask_area,
			hole_area=hole_area,
			rect=rect,
			centroid=centroid,
			hull_area=hull_area,
			hull_perimeter=hull_perimeter,
			huecos=int(huecos),
			hole_area_ratio=hole_area_ratio,
			circularity=circularity,
			solidity=solidity,
			inertia_ratio=inertia_ratio,
			r_hull=r_hull,
			inner_gradient=inner_gradient,
			radiar_var=radial_var,
			contour_source="mask" if contour_from_mask else "meta",
		)

		vec_values = [
			float(huecos),
			float(r_hull),
			float(radial_var)
		]

		if self.mode == "5D":
			vec_values.extend([float(circularity), float(solidity), float(inertia_ratio)])

		vec = np.asarray(vec_values, dtype=np.float32)
		return vec, names, debug

	# ------------------------------------------------------------------ #
	@staticmethod
	def _2mask(
		mask: np.ndarray
		) -> Mask:
		"""Asegura que la máscara es uint8 {0,255}."""

		if mask is None:
			return None
		m = np.asarray(mask)
		if m.dtype != np.uint8:
			m = (m > 0).astype(np.uint8)
		if m.max(initial=0) <= 1:
			m = m * 255
		else:
			m = np.where(m > 0, 255, 0).astype(np.uint8)
		return m
	
	def _variacion_radial(self, contour: np.ndarray, centroid: tuple[float, float]) -> float:
		"""
		Mide cuán constante es la distancia del contorno al centro.
		Devuelve std(r) / mean(r). 
		- Cercano a 0  -> contorno casi circular
		- Más grande   -> contorno poligonal/irregular
		"""
		if contour is None or len(contour) < 3:
			return 0.0

		cx, cy = centroid

		# contour típico de OpenCV: (N, 1, 2). Se aplasta a (N, 2)
		pts = contour.reshape(-1, 2).astype(np.float32)

		xs = pts[:, 0]
		ys = pts[:, 1]

		rs = np.sqrt((xs - cx)**2 + (ys - cy)**2)

		r_mean = float(rs.mean())
		r_std  = float(rs.std())

		if r_mean <= 1e-6:
			return 0.0

		radial_var = r_std / r_mean

		radial_var = min(max(radial_var, self.hp.radial_var_t_low), self.hp.radial_var_t_high)
		m = 1.0 / (self.hp.radial_var_t_high)
		radial_var = m*radial_var

		return radial_var

	@staticmethod
	def _contorno_jerarquia(
		mask: Mask,
	) -> tuple[np.ndarray | None, np.ndarray, list[int]]:
		"""
		Obtiene el contorno externo de mayor área y la jerarquía CCOMP.

		Returns
		-------
		contour : np.ndarray | None
			Contorno principal.
		hierarchy : np.ndarray
			Jerarquía completa (shape (N,4)).
		hole_indices : list[int]
			Índices de los hijos directos del contorno principal.
		"""

		cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
		if not cnts:
			return None, np.zeros((0, 4), dtype=np.int32), []

		areas = [cv2.contourArea(c) for c in cnts]
		idx = int(np.argmax(areas))
		contour = cnts[idx]

		if hier is None or len(hier) == 0:
			return contour, np.zeros((len(cnts), 4), dtype=np.int32), []

		hier = hier[0]
		hole_indices: list[int] = []
		child = hier[idx][2]
		while child != -1:
			hole_indices.append(child)
			child = hier[child][0]
		return contour, hier, hole_indices

	@staticmethod
	def _contorno_area_y_perimetro(
		contour: np.ndarray
		) -> tuple[float, float]:
		"""Calcula área y perímetro (arcLength) para un contorno."""

		area = float(cv2.contourArea(contour))
		perimetro = float(cv2.arcLength(contour, True))
		return area, perimetro

	@staticmethod
	def _hull_convexo_y_perimetro(
		contour: np.ndarray
		) -> tuple[np.ndarray | None, float]:
		"""Obtiene el casco convexo y su perímetro."""

		hull = cv2.convexHull(contour) if contour is not None else None
		if hull is None or len(hull) == 0:
			return None, 0.0
		hull_perimeter = float(cv2.arcLength(hull, True))
		return hull, hull_perimeter

	@staticmethod
	def _inercia_desde_momentos(contour: np.ndarray) -> float:
		"""Deriva λmin/λmax a partir de los momentos centrales."""

		M = cv2.moments(contour)
		m00 = M["m00"] + 1e-9
		cov_xx = M["mu20"] / m00
		cov_yy = M["mu02"] / m00
		cov_xy = M["mu11"] / m00
		cov = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]], dtype=np.float64)
		eigvals, _ = np.linalg.eig(cov + 1e-9 * np.eye(2, dtype=np.float64))
		eigvals = np.sort(eigvals)
		return float(eigvals[0] / (eigvals[1] + 1e-9))

	@staticmethod
	def _gradiente_interno(img_norm: np.ndarray, mask: np.ndarray) -> float:
		"""Calcula la mediana del gradiente (Sobel) dentro de la máscara erosionada."""

		if img_norm.ndim == 3:
			img_gray = cv2.cvtColor(img_norm, cv2.COLOR_BGR2GRAY)
		else:
			img_gray = img_norm

		img_gray = np.asarray(img_gray, dtype=np.float32, copy=False)
		kernel = np.ones((3, 3), dtype=np.uint8)
		eroded = cv2.erode(mask, kernel, iterations=1)
		if cv2.countNonZero(eroded) == 0:
			eroded = mask

		gx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
		gy = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
		mag = cv2.magnitude(gx, gy)

		values = mag[eroded > 0]
		if values.size == 0:
			return 0.0
		return float(np.median(values))
