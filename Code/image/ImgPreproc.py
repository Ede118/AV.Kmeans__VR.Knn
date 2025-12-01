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

    cfg: ImgPreprocCfg = field(default_factory=ImgPreprocCfg)

    # ------------------------------------------------------------------ #
    # API pública
    # ------------------------------------------------------------------ #
    def process(
        self, 
        img_color: ImgColorF,
        blacknwhite: bool = False
        ) -> "ImgPreproc":
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
            img_sq = cv.cvtColor(img_sq, cv.COLOR_BGR2GRAY).astype(np.float32) / 255.0


        return img_sq, mask_sq

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

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray_blur = cv.GaussianBlur(src=gray, ksize=(0, 0), sigmaX=self.cfg.sigma)

        # Aplicar Otsu
        _, mask = cv.threshold(
            gray_blur,
            0,
            255,
            cv.THRESH_BINARY_INV + cv.THRESH_OTSU
        )

        return mask


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
        
        if open_ksize % 2 != 1 and close_ksize % 2 != 1:
            raise ValueError("El kernel debe tener tamaño impar.")

        kernel_open = np.ones((open_ksize, open_ksize), np.uint8)
        kernel_close = np.ones((close_ksize, close_ksize), np.uint8)

        mask_clean = cv.morphologyEx(mask_sq, cv.MORPH_OPEN, kernel_open, iterations=1)
        mask_clean = cv.morphologyEx(mask_clean, cv.MORPH_CLOSE, kernel_close, iterations=1)
        
        return mask_clean
