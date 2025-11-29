from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

import cv2 as cv
import numpy as np

from Code import VecF, MatF, ImgColorF, ImgGrayF, Mask, ImgColor, ImgGray


@dataclass(slots=True)
class SegMeta:
    """Metadata geométrica resultante de la segmentación y el recorte."""

    contour: np.ndarray
    rect: tuple
    centroid: tuple[float, float]
    inertia_ratio: float
    aspect_ratio: float
    circularity: float
    holes: int
    M_warp: np.ndarray

@dataclass
class BackgroundModel:
    """Modelo de fondo: estadísticos globales en HSV."""
    stats_path: Path                    
    bg_dir: Path                        
    stats: dict = field(default_factory=dict) 

    def compute_stats(self) -> None:
        """Calcula estadísticos globales H y V a partir de todas las imágenes de bg_dir."""
        paths = sorted(self.bg_dir.rglob("*.jpg"))
        if not paths:
            raise ValueError(f"No se encontraron imágenes de background en {self.bg_dir}")

        H_all = []
        V_all = []

        for p in paths:
            img = cv.imread(str(p))
            if img is None:
                continue
            hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
            H = hsv[:, :, 0].astype(np.float32)
            V = hsv[:, :, 2].astype(np.float32)

            H_all.append(H.ravel())
            V_all.append(V.ravel())

        if not H_all:
            raise ValueError("No se pudo cargar ninguna imagen de background válida.")

        H_all = np.concatenate(H_all)
        V_all = np.concatenate(V_all)

        # Rango típico de H (evitando outliers)
        H_min = np.percentile(H_all, 5)
        H_max = np.percentile(H_all, 95)

        # Saturación mínima útil (por si querés luego filtrar lavados)
        # S_all = ... si lo necesitás más adelante

        V_mean = float(np.mean(V_all))
        V_std = float(np.std(V_all))

        self.stats = {
            "H_min": H_min,
            "H_max": H_max,
            "V_mean": V_mean,
            "V_std": V_std,
        }

    def save(self) -> None:
        """Guarda estadísticos en un .npz comprimido."""
        self.stats_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.stats:
            raise ValueError("No hay estadísticos calculados para guardar.")
        np.savez_compressed(self.stats_path, **self.stats)

    def load(self) -> None:
        """Carga estadísticos desde un .npz."""
        data = np.load(self.stats_path, allow_pickle=False)
        self.stats = {k: data[k].item() if data[k].shape == () else data[k] for k in data.files}

        

        
@dataclass(slots=True)
class PreprocOutput:
    """Salida canónica del preprocesamiento de imágenes."""

    img: np.ndarray
    mask: MaskU8
    meta: Optional[SegMeta] | None = None


@dataclass(slots=True)
class ImgPreprocCfg:
    """
    Configuración del pipeline de preprocesamiento.

    La intención es replicar la lógica de `tests/test_image_ImgPreproc.py`
    removiendo cualquier preocupación de visualización.
    """

    target_size: int = 256
    sigma: float = 3.0

    flag_BnW: bool = False
    
    flag_refine_mask: bool = False
    open_ksize: int = 3
    close_ksize: int = 3
    


@dataclass(slots=True)
class ImgPreproc:
    """
    Pipeline de preprocesamiento geométrico y fotométrico.

    - Normaliza iluminación.
    - Segmenta el objeto dominante.
    - Estima geometría para un recorte alineado.
    - Devuelve imagen y máscara ya redimensionadas a `target_size`.
    """

    bg_dir: Path
    bg_stats_path: Path
    bg_model: BackgroundModel = field(init=False)

    cfg: ImgPreprocCfg = field(default_factory=ImgPreprocCfg)
    
    def __post_init__(self) -> None:
        self.bg_model = BackgroundModel(
            stats_path=self.bg_stats_path,
            bg_dir=self.bg_dir,
        )

        if self.bg_stats_path.exists():
            # Cargamos modelo ya calculado
            self.bg_model.load()
        else:
            # Calculamos una sola vez y guardamos
            self.bg_model.compute_stats()
            self.bg_model.save()

    # ------------------------------------------------------------------ #
    # API pública
    # ------------------------------------------------------------------ #
    def process(
        self, 
        img_color: ImgColorF,
        blacknwhite: bool = False
        ) -> PreprocOutput:
        """
        Ejecuta el pipeline completo sobre una imagen BGR/Gray.

        Devuelve `PreprocOutput` con:
        - `img`   : float32 en [0, 1], tamaño `cfg.target_size`.
        - `mask`  : uint8 {0,255}, alineada con `img`.
        - `meta`  : detalles geométricos del objeto detectado (o `None`).
        """

        mask_obj = self._normalize(img_color)
        img_sq, mask_sq = self._crop_and_square(img_color, mask_obj, size=self.cfg.target_size)
        
        if self.cfg.flag_refine_mask:
            mask_sq = self._refine_mask(mask_sq, open_ksize=self.cfg.open_ksize, close_ksize=self.cfg.close_ksize)

        if blacknwhite:
            # Pasar a gris float32 normalizado
            img_sq = cv.cvtColor(img_sq, cv.COLOR_BGR2GRAY).astype(np.float32) / 255.0


        return PreprocOutput(img=img_sq, mask=mask_sq)

    # ------------------------------------------------------------------ #
    # Helpers privados
    # ------------------------------------------------------------------ #
    def _normalize(
        self,
        img: ImgColorF,
        ) -> Mask:
        """
        Segmenta el objeto usando:
        - color de fondo (rango típico de H),
        - diferencia de brillo respecto al modelo de fondo (V_mean, V_std),
        y luego se queda con las componentes centrales más grandes.
        """
        if not self.bg_model.stats:
            raise RuntimeError("BackgroundModel no tiene estadísticas cargadas.")

        H_min = self.bg_model.stats["H_min"]
        H_max = self.bg_model.stats["H_max"]
        V_mean = self.bg_model.stats["V_mean"]
        V_std = self.bg_model.stats["V_std"]

        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        hsv_blur = cv.GaussianBlur(hsv, (0, 0), sigmaX=self.cfg.sigma)

        H = hsv_blur[:, :, 0].astype(np.float32)
        S = hsv_blur[:, :, 1].astype(np.float32)
        V = hsv_blur[:, :, 2].astype(np.float32)

        dh = 5.0
        lower_H = max(0.0, H_min - dh)
        upper_H = min(179.0, H_max + dh)

        S_min = 30.0

        # Máscara de fondo por color: H en rango y S suficientemente alta
        mask_bg_color = np.zeros_like(V, dtype=np.uint8)
        mask_bg_color[
            (H >= lower_H) & (H <= upper_H) & (S >= S_min)
        ] = 255

        if V_std < 1e-3:
            V_std = 1.0  # evitar división por algo ridículo

        k = 2.0  # cuántas desviaciones permitimos
        diff_V = np.abs(V - V_mean)
        mask_bg_brightness = np.zeros_like(V, dtype=np.uint8)
        mask_bg_brightness[diff_V <= k * V_std] = 255


        mask_bg = cv.bitwise_and(mask_bg_color, mask_bg_brightness)

        mask_obj = cv.bitwise_not(mask_bg)

        # mask_obj = self._keep_center_components(mask_obj)

        return mask_obj


    def _bbox_from_mask(
        self,
        mask: Mask
        ) -> tuple[int, int, int, int]:
        ys, xs = np.where(mask>0)
        if xs.size == 0 or ys.size == 0:
            raise ValueError("Máscara vacía.")
        
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        
        return x1, y1, x2, y2

    def _expand_bbox(
        self,
        x1, 
        y1, 
        x2, 
        y2, 
        img_shape, 
        margin=0.10):
        
        h, w = img_shape[:2]

        bw = x2 - x1
        bh = y2 - y1

        # agrandamos ancho/alto
        extra_w = int(bw * margin / 2)
        extra_h = int(bh * margin / 2)

        x1 = max(0, x1 - extra_w)
        x2 = min(w, x2 + extra_w)
        y1 = max(0, y1 - extra_h)
        y2 = min(h, y2 + extra_h)

        return x1, y1, x2, y2

    def _crop_and_square(
        self,
        img: ImgColor,
        mask: Mask,
        size: int = 256,
        ) -> tuple[ImgColor, Mask]:

        x1, y1, x2, y2 = self._bbox_from_mask(mask)
        x1, y1, x2, y2 = self._expand_bbox(x1, y1, x2, y2, img.shape, margin=0.10)

        # Recorte
        img_crop = img[y1:y2, x1:x2]
        mask_crop = mask[y1:y2, x1:x2]

        h, w = img_crop.shape[:2]
        scale = size / max(h, w)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        # Resize manteniendo aspecto
        img_resized = cv.resize(img_crop, (new_w, new_h), interpolation=cv.INTER_AREA)
        mask_resized = cv.resize(mask_crop, (new_w, new_h), interpolation=cv.INTER_NEAREST)

        # Lienzo cuadrado con fondo negro (o verde, o lo que quieras)
        img_sq = np.zeros((size, size, 3), dtype=img.dtype)
        mask_sq = np.zeros((size, size), dtype=mask.dtype)

        # Centrado
        y_off = (size - new_h) // 2
        x_off = (size - new_w) // 2

        img_sq[y_off:y_off+new_h, x_off:x_off+new_w] = img_resized
        mask_sq[y_off:y_off+new_h, x_off:x_off+new_w] = mask_resized

        return img_sq, mask_sq

    def _refine_mask(
        self, 
        mask_sq: Mask,
        open_ksize: int = 3,
        close_ksize: int = 3
        ) -> Mask:
        
        if open_ksize // 2 == 0 and close_ksize // 2 == 0:
            raise ValueError("El kernel debe tener tamaño impar.")

        kernel_open = np.ones((open_ksize, open_ksize), np.uint8)
        kernel_close = np.ones((close_ksize, close_ksize), np.uint8)

        mask_clean = cv.morphologyEx(mask_sq, cv.MORPH_OPEN, kernel_open, iterations=1)
        mask_clean = cv.morphologyEx(mask_clean, cv.MORPH_CLOSE, kernel_close, iterations=1)
        
        return mask_clean
