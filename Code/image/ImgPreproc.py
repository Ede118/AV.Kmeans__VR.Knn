from dataclasses import dataclass, field
from typing import Tuple, Literal
import cv2
import numpy as np

NormKind = Literal['none','zero_one','zscore']

from Code.types import ColorImageU8, GrayImageF32, MaskU8


# ---  A) Config   --- #
@dataclass(slots=True)
class ImgPreprocCfg:
    target_size: Tuple[int, int] = (128, 128)  # (H, W)
    keep_aspect: bool = True                   # respeta aspecto y hace pad centrado
    to_gray: bool = True                       # BGR --> Gray
    normalize_kind: NormKind = 'zero_one'      # fotometría: none | zero_one | zscore
    blur_ksize: int = 0                        # 0 = sin blur; si >1, se fuerza impar
    open_ksize: int = 3                        # > 0 --> no aplicar apertura
    close_ksize: int = 0                       # 0 --> no rellenar huecos
    seg_min_area: float = 0.002                  # área mínima en px para aceptar un blob
    crop_to_bbox: bool = False
    crop_margin: int = 2

# ---  B) Procesador (conducta)  --- #

@dataclass(slots=True)
class ImgPreproc:
    # Constructor default: crea el objeto con las configuraciones del objeto "PreprocCfg"
    cfg: ImgPreprocCfg = field(default_factory=ImgPreprocCfg)


	# -------------------------------------------------------------------------------------------------  #

    def process(self,
            img_bgr: ColorImageU8, 
            *, 
            float_mask: bool = True
            ) -> tuple[GrayImageF32, MaskU8]:
        """
        ### Método principal
        Se inserta una un array y devuelve una tupla
        - Imagen en grises normalizada (reshape, cut borders) 
        - Máscara de [0..255] 
        ### Resumen
        
        ```
        pre = ImgPreproc()
        paths: Path = getPath() #Funcion ficticia
        img = cv2.imread(p)
        Imagen_gris, Mascara = pre.process(img)
        ```
        """
        img = self.normalize(img_bgr)
        mask = self.segment(img)                    # uint8 {0,255}
        if float_mask:
            mask = (mask > 0).astype(np.float32)    # {0,1} float32
        return img, mask

    # --------- API pública --------- #
    def normalize(
        self, 
        img_bgr: ColorImageU8
        ) -> GrayImageF32:
        """
        ### Normalización geométrica/fotométrica
        Convierte una imagen BGR/Gray a formato canónico para features.
        - Convierte a gris si `cfg.to_gray`
        - Redimensiona y aplica pad según `cfg.target_size` y `cfg.keep_aspect`
        - Aplica blur gaussiano si `cfg.blur_ksize`
        - Escala valores según `cfg.normalize_kind`
        ### Resumen

        ```
        pre = ImgPreproc()
        img_norm = pre.normalize(img_bgr)
        ```
        """
        x = img_bgr

        # 1) Se transforma en gris dependiendo de la configuración.
        # Depende de: self.cfg.to_gray ? true : false
        if self.cfg.to_gray and x.ndim == 3:
            # Función de cv2
            x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)

        # 2) Redimensionar a tamaño objetivo
        # En self.cfg.target_size es el tamaño que se debe tener, 128 x 128 default
        # Si se quiere mantener aspecto según self.cfg.keep_aspect ? true : false
        x = self._resize_pad(x, self.cfg.target_size, self.cfg.keep_aspect)

        # 3) Blur opcional para bajar ruido de alta freq
        # Depende de (self.cfg.blur_ksize != 0) ? true : false 
        if self.cfg.blur_ksize and self.cfg.blur_ksize > 1:
            k = self.cfg.blur_ksize | 1  # aseguramos impar
            x = cv2.GaussianBlur(x, (k, k), 0)

        # 4) Normalización fotométrica
        x = self._normalize_values(x, self.cfg.normalize_kind)

        return x.astype(np.float32)

    # -------------------------------------------------------------------------------------------------  #

    def segment(self, img_norm: GrayImageF32) -> MaskU8:
        """
        ### Segmentación del objeto
        Genera máscara binaria del objeto principal a partir de una imagen normalizada.
        - Umbral Otsu con fallback adaptativo
        - Ajusta polaridad para dejar fondo oscuro
        - Opcionalmente aplica apertura/cierre según cfg
        - Conserva la componente conexa mayor y filtra por área mínima
        ### Resumen

        ```
        pre = ImgPreproc()
        mask = pre.segment(img_norm)
        ```
        """

        # Asegurar uint8 de 0..255 para umbralizar
        x8 = self._as_u8(img_norm)

        # 1) Umbral global (Otsu) con fallback adaptativo si salió trivial
        _, th = cv2.threshold(x8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if th.sum() in (0, 255 * th.size):
            th = cv2.adaptiveThreshold(
                x8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 35, 5
            )

        # 2) Asegurar polaridad: objeto en blanco (fondo tiende a negro)
        if self._border_is_white(th):
            th = cv2.bitwise_not(th)

        # 3) Ver si rellenar los huecos o no.
        k = self.cfg.open_ksize
        if k and k > 1:
            th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((k, k), np.uint8), 1)

        k = self.cfg.close_ksize
        if k and k > 1:
            th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8), 1)

        # 4) Elegir componente conexa principal
        mask = self._largest_component(th)

        # 5) Filtro por área mínima
        H, W = mask.shape
        min_area = max(1, int(self.cfg.seg_min_area * H * W))

        if cv2.countNonZero(mask) < min_area:
            mask[:] = 0

        return mask

    # -------------------------------------------------------------------------------------------------  #

    def meta(self) -> dict:
        """Pequeña trazabilidad del preproc para auditar runs."""
        return {
            'size': self.cfg.target_size,
            'keep_aspect': self.cfg.keep_aspect,
            'to_gray': self.cfg.to_gray,
            'normalize': self.cfg.normalize_kind,
            'blur_ksize': self.cfg.blur_ksize,

        }

    # -------------------------------------------------------------------------------------------------  #
    #                               --------- Helpers privados  ---------                                #
    # -------------------------------------------------------------------------------------------------  #
    def _resize_pad(
            self, 
            x: np.ndarray, 
            target: Tuple[int, int], 
            keep_aspect: bool
            ) -> np.ndarray:
        """
        ### Redimensionado con padding
        Ajusta la imagen al tamaño objetivo manteniendo aspecto si se solicita.
        - Si `keep_aspect`, hace resize proporcional y completa con bordes negros centrados
        - Si no, hace resize directo al tamaño target
        ### Resumen

        ```
        pre = ImgPreproc()
        resized = pre._resize_pad(img, (128, 128), keep_aspect=True)
        ```
        """
        
        H, W = target
        if keep_aspect:
            ih, iw = x.shape[:2]
            s = min(H / ih, W / iw)
            nh, nw = max(1, int(ih * s)), max(1, int(iw * s))
            x = cv2.resize(x, (nw, nh), interpolation=cv2.INTER_AREA)
            top  = (H - nh) // 2
            left = (W - nw) // 2
            bot  = H - nh - top
            right= W - nw - left
            padval = 0
            x = cv2.copyMakeBorder(x, top, bot, left, right, cv2.BORDER_CONSTANT, value=padval)
            return x
        else:
            return cv2.resize(x, (W, H), interpolation=cv2.INTER_AREA)

    # -------------------------------------------------------------------------------------------------  #

    def _normalize_values(self, x: np.ndarray, kind: NormKind) -> np.ndarray:
        """
        ### Escalado fotométrico
        Normaliza valores de intensidad según la política elegida.
        - `none`: deja valores originales
        - `zero_one`: usa percentiles 1-99 para escalar a [0,1]
        - `zscore`: centra y escala por desviación estándar
        ### Resumen

        ```
        pre = ImgPreproc()
        x = pre._normalize_values(img, "zero_one")
        ```
        """
        
        x = x.astype(np.float32)
        if kind == 'none':
            return x
        if kind == 'zero_one':
            p1, p99 = np.percentile(x, 1), np.percentile(x, 99)
            if not np.isfinite(p1) or not np.isfinite(p99) or p99 <= p1:
                p1, p99 = float(x.min()), float(x.max())
            return np.clip((x - p1) / (p99 - p1 + 1e-6), 0, 1)
        # zscore
        mu, sd = x.mean(), x.std()
        return (x - mu) / (sd + 1e-6)

    # -------------------------------------------------------------------------------------------------  #

    def _as_u8(self, x: np.ndarray) -> np.ndarray:
        """
        ### Conversión a uint8
        Convierte arrays float (0-1 o z-score) a `uint8` [0,255] antes de umbralizar.
        - Si ya es `uint8`, lo devuelve tal cual
        - Usa percentiles para z-score u otros rangos
        ### Resumen

        ```
        pre = ImgPreproc()
        x8 = pre._as_u8(img_norm)
        ```
        """

        
        if x.dtype == np.uint8:
            return x
        xmax = float(x.max())
        xmin = float(x.min())
        
        if xmax <= 1.5 and xmin >= -0.1:
            return np.clip(x * 255.0, 0, 255).astype(np.uint8)
        
        # zscore u otros rangos: normalizamos robusto
        p1, p99 = np.percentile(x, 1), np.percentile(x, 99)
        x = np.clip((x - p1) / (p99 - p1 + 1e-6), 0, 1)
        return (x * 255.0).astype(np.uint8)

    # -------------------------------------------------------------------------------------------------  #

    def _border_is_white(self, th: np.ndarray) -> bool:
        """
        ### Detección de fondo claro
        Calcula la media del borde de la máscara para decidir si hay que invertirla.
        - Útil para asegurar objeto en blanco y fondo oscuro
        ### Resumen

        ```
        pre = ImgPreproc()
        invertir = pre._border_is_white(mask)
        ```
        """

        
        border = np.concatenate([th[0, :], th[-1, :], th[:, 0], th[:, -1]])
        return border.mean() > 127

    # -------------------------------------------------------------------------------------------------  #

    def _largest_component(self, th: np.ndarray) -> MaskU8:
        """
        ### Selección de componente principal
        Mantiene solo el blob más grande tras la segmentación.
        - Usa `connectedComponentsWithStats` y descarta el resto
        ### Resumen

        ```
        pre = ImgPreproc()
        main_mask = pre._largest_component(mask)
        ```
        """
        
        num, labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
        
        if num <= 1:
            return th
        
        areas = stats[1:, cv2.CC_STAT_AREA]
        idx = 1 + int(np.argmax(areas)) if len(areas) else 0
        mask = (labels == idx).astype(np.uint8) * 255
        
        return mask

    # ------------------------------------------------------------------------------------------------- #

    def _crop_to_mask(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        *,
        margin: int = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        if cv2.countNonZero(mask) == 0:
            return img, mask
        x, y, w, h = cv2.boundingRect(mask)
        x0 = max(0, x - margin)
        y0 = max(0, y - margin)
        x1 = min(img.shape[1], x + w + margin)
        y1 = min(img.shape[0], y + h + margin)
        return img[y0:y1, x0:x1], mask[y0:y1, x0:x1]

