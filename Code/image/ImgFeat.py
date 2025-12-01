from __future__ import annotations
from dataclasses import dataclass

from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from Code.image.ImgPreproc import SegMeta  # pragma: no cover

@dataclass(init=False)
class ImgFeat:
    """Extractor de descriptores geométricos/fotométricos para objetos segmentados."""

    _VALID_MODES = {"3D": 3, "6D": 6}

    def __init__(self, mode: str = "3D", use_meta: bool = False) -> None:
        self.mode = mode.upper()
        if self.mode not in self._VALID_MODES:
            raise ValueError(f"mode debe ser '3D' o '6D', no '{mode}'")
        self.use_meta = bool(use_meta)

    # ------------------------------------------------------------------ #
    @staticmethod
    def feature_names(mode: str = "3D") -> List[str]:
        """Devuelve los nombres de features en el orden exacto del vector."""

        mode_norm = mode.upper()
        if mode_norm == "3D":
            return [
                "n_holes",
                "r_hull",
                "radiar_var"
            ]
        if mode_norm == "6D":
            return [
                "n_holes",
                "r_hull",
                "radiar_var",
                "circularity",
                "solidity",
                "inertia_ratio",
            ]
        raise ValueError(f"mode debe ser '3D' o '5D', no '{mode}'")

    # ------------------------------------------------------------------ #
    def extract(
        self,
        img_norm: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, List[str], Dict[str, object]]:
        """
        Calcula el vector de features y metadatos de apoyo.

        Parameters
        ----------
        img_norm : np.ndarray
            Imagen float32 ∈ [0,1] alineada y centrada (salida de ImgPreproc).
        mask : np.ndarray
            Máscara uint8 {0,255} (o convertible) alineada con `img_norm`.
        meta : SegMeta | None
            Información opcional proveniente de ImgPreproc (contorno, rect, etc.).

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
        mask_u8 = self._ensure_mask_uint8(mask)
        debug: Dict[str, object] = {}

        if mask_u8 is None or cv2.countNonZero(mask_u8) == 0:
            debug["empty_mask"] = True
            return np.zeros(n_dim, dtype=np.float32), names, debug

        img_f32 = np.asarray(img_norm, dtype=np.float32)

        contour = None
        rect = None
        inertia_ratio = None
        holes_meta = None

        contour_from_mask = False
        hole_indices: List[int] = []

        if contour is None:
            contour, hierarchy, hole_indices = self._main_contour_and_hierarchy(mask_u8)
            contour_from_mask = True
            if contour is None or len(contour) < 3:
                debug["empty_contour"] = True
                return np.zeros(n_dim, dtype=np.float32), names, debug
        else:
            hierarchy = None  # reutilizaremos la máscara para métricas de huecos

        area, perimeter = self._contour_area_and_perimeter(contour)
        mask_area = float(cv2.countNonZero(mask_u8))
        hull, hull_perimeter = self._convex_hull_and_perimeter(contour)
        hull_area = float(cv2.contourArea(hull)) if hull is not None else 0.0

        if rect is None:
            rect = cv2.minAreaRect(contour)

        centroid = tuple(rect[0]) if rect else (0.0, 0.0)
        radial_var = self._radial_variation(contour, centroid)

        if inertia_ratio is None:
            inertia_ratio = self._inertia_ratio_from_moments(contour)

        if holes_meta is not None:
            n_holes = int(holes_meta)
        elif hierarchy is not None:
            n_holes = len(hole_indices)
        else:
            # meta sin jerarquía: aproximar desde máscara rellenando huecos
            contour_tmp, hierarchy_tmp, hole_indices = self._main_contour_and_hierarchy(mask_u8)
            n_holes = len(hole_indices)
            if contour is None and contour_tmp is not None:
                contour = contour_tmp
                area, perimeter = self._contour_area_and_perimeter(contour)
                hull, hull_perimeter = self._convex_hull_and_perimeter(contour)
                hull_area = float(cv2.contourArea(hull)) if hull is not None else 0.0
                rect = cv2.minAreaRect(contour)

        hueco = 1 if n_holes >= 1 else 0

        hole_area = max(area - mask_area, 0.0)
        hole_area_ratio = hole_area / (area + 1e-9)

        circularity = (4.0 * np.pi * area) / (perimeter * perimeter + 1e-9)
        solidity = area / (hull_area + 1e-9) if hull_area > 0 else 0.0
        
        r_hull = ((perimeter / (hull_perimeter + 1e-9)) - 1.0) * 3
        

        inner_gradient = self._inner_gradient(img_f32, mask_u8)

        debug.update(
            area=area,
            perimeter=perimeter,
            mask_area=mask_area,
            hole_area=hole_area,
            rect=rect,
            centroid=centroid,
            hull_area=hull_area,
            hull_perimeter=hull_perimeter,
            n_holes=int(n_holes),
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
            float(hueco),
            float(r_hull),
            float(radial_var)
        ]

        if self.mode == "5D":
            vec_values.extend([float(circularity), float(solidity), float(inertia_ratio)])

        vec = np.asarray(vec_values, dtype=np.float32)
        return vec, names, debug

    # ------------------------------------------------------------------ #
    @staticmethod
    def _ensure_mask_uint8(mask: np.ndarray) -> np.ndarray:
        """Convierte la máscara a uint8 {0,255}."""

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

    @staticmethod
    def _main_contour_and_hierarchy(
        mask: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], np.ndarray, List[int]]:
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
        hole_indices: List[int] = []
        child = hier[idx][2]
        while child != -1:
            hole_indices.append(child)
            child = hier[child][0]
        return contour, hier, hole_indices

    @staticmethod
    def _contour_area_and_perimeter(contour: np.ndarray) -> Tuple[float, float]:
        """Calcula área y perímetro (arcLength) para un contorno."""

        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        return area, perimeter

    @staticmethod
    def _convex_hull_and_perimeter(contour: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """Obtiene el casco convexo y su perímetro."""

        hull = cv2.convexHull(contour) if contour is not None else None
        if hull is None or len(hull) == 0:
            return None, 0.0
        hull_perimeter = float(cv2.arcLength(hull, True))
        return hull, hull_perimeter

    @staticmethod
    def _inertia_ratio_from_moments(contour: np.ndarray) -> float:
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
    def _inner_gradient(img_norm: np.ndarray, mask: np.ndarray) -> float:
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

    @staticmethod
    def _radial_variation(contour: np.ndarray, centroid: tuple[float, float]) -> float:
        """
        Mide cuán constante es la distancia del contorno al centro.
        Devuelve std(r) / mean(r). 
        - Cercano a 0  -> contorno casi circular
        - Más grande   -> contorno poligonal/irregular
        """
        if contour is None or len(contour) < 3:
            return 0.0

        cx, cy = centroid

        # contour típico de OpenCV: (N, 1, 2). Lo aplastamos a (N, 2)
        pts = contour.reshape(-1, 2).astype(np.float32)

        xs = pts[:, 0]
        ys = pts[:, 1]

        rs = np.sqrt((xs - cx)**2 + (ys - cy)**2)

        r_mean = float(rs.mean())
        r_std  = float(rs.std())

        if r_mean <= 1e-6:
            return 0.0

        radial_var = r_std / r_mean

        radial_var = min(max(radial_var, 0), 0.05)
        m = 20.0
        radial_var = m*radial_var

        return radial_var