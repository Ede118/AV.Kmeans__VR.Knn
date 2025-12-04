from __future__ import annotations

from dataclasses import dataclass, field

import cv2 as cv
import numpy as np

from Code import VecF, MatF, Mask, ImgColor, ImgGray



@dataclass(slots=True)
class ImgPreprocCfg:
	"""
	Configuración del pipeline de preprocesamiento.

	La intención es replicar la lógica de `tests/test_image_ImgPreproc.py`
	removiendo cualquier preocupación de visualización.
	"""

	target_size: int = 256
	sigma: float = 3.0
	margin: float = 0.10

	flag_refine_mask: bool = False
	open_ksize: int = 3
	close_ksize: int = 3
	
	flag_BnW: bool = False


@dataclass(slots=True)
class ImgPreproc:
	"""
	Pipeline de preprocesamiento geométrico y fotométrico.

	- Normaliza iluminación.
	- Segmenta el objeto dominante.
	- Estima geometría para un recorte alineado.
	- Devuelve imagen y máscara ya redimensionadas a `target_size`.
	"""

	config: ImgPreprocCfg = field(default_factory=ImgPreprocCfg)

	# ------------------------------------------------------------------ #
	# API pública
	# ------------------------------------------------------------------ #
	def procesar(
		self, 
		img_color: ImgColor,
		blacknwhite: bool = False
		) -> "ImgPreproc":
		"""
		Ejecuta el pipeline completo sobre una imagen BGR/Gray.

		Devuelve `PreprocOutput` con:
		- `img`   : float32 en [0, 1], tamaño `config.target_size`.
		- `mask`  : uint8 {0,255}, alineada con `img`.
		- `meta`  : detalles geométricos del objeto detectado (o `None`).
		"""

		mask_obj = self._normalizar(img_color)
		
		img_sq, mask_sq = self._recortar(img_color, mask_obj)
		
		if self.config.flag_refine_mask:
			mask_sq = self._refinar_mask(mask_sq)

		if blacknwhite:
			img_sq = cv.cvtColor(img_sq, cv.COLOR_BGR2GRAY).astype(np.float32) / 255.0


		return img_sq, mask_sq

	# ------------------------------------------------------------------ #
	# Helpers privados
	# ------------------------------------------------------------------ #
	def _normalizar(
		self,
		img: ImgColor,
		) -> Mask:
		"""
		Segmenta el objeto dominante de la imagen usando umbralización de Otsu.
		Aplica los siguientes pasos:
		1. Convierte la imagen BGR a escala de grises.
		2. Aplica desenfoque gaussiano con sigma configurado.
		3. Aplica umbralización de Otsu inversa (THRESH_BINARY_INV) para obtener máscara binaria.
		
		Args:
		img: Imagen de entrada en formato BGR/Color (float32).
		Returns:
		Máscara binaria (uint8) donde píxeles del objeto = 255, fondo = 0.
		
		Raises:
		ValueError: Si la imagen de entrada es inválida.

		Ejemplo:
		```
		mask = preproc._normalize(img_color)
		``` 
		"""
		if img is None or img.size == 0:
			raise ValueError("La imagen de entrada está vacía.")

		gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
		gray_blur = cv.GaussianBlur(src=gray, ksize=(0, 0), sigmaX=self.config.sigma)

		# Aplicar Otsu
		_, mask = cv.threshold(
			gray_blur,
			0,
			255,
			cv.THRESH_BINARY_INV + cv.THRESH_OTSU
		)

		return mask


	def _bbox_desde_mask(
		self,
		mask: Mask
		) -> tuple[int, int, int, int]:
		"""
		Extrae el bounding box mínimo que contiene todos los píxeles de la máscara.
		Encuentra las coordenadas de la región rectangular más pequeña que envuelve
		todos los píxeles no nulos en la máscara.
		
		Args:
		mask: Máscara binaria (uint8) donde píxeles > 0 pertenecen al objeto.
		
		Returns:
		Tupla (x1, y1, x2, y2) con las coordenadas de la esquina superior-izquierda
		e inferior-derecha del bounding box.
		
		Raises:
		ValueError: Si la máscara está vacía (sin píxeles > 0).
		ValueError: Si la máscara no es 2D o no es de tipo entero.
		Ejemplo:
		```
		x1, y1, x2, y2 = preproc._bbox_from_mask(mask)
		```
		"""
		if mask is None or mask.size == 0:
			raise ValueError("La máscara de entrada está vacía.")
		if mask .ndim != 2:
			raise ValueError("La máscara debe ser 2D.")
		if not np.issubdtype(mask.dtype, np.integer):
			raise ValueError("La máscara debe tener un tipo de dato entero.")

		ys, xs = np.where(mask>0)
		if xs.size == 0 or ys.size == 0:
			raise ValueError("Máscara vacía.")
		
		x1, x2 = xs.min(), xs.max()
		y1, y2 = ys.min(), ys.max()
		
		return x1, y1, x2, y2

	def _expandir_bbox(
		self,
		x1, 
		y1, 
		x2, 
		y2, 
		img_shape
		) -> tuple[int, int, int, int]:
		"""
		Expande el bounding box por un margen porcentual respetando los límites de la imagen.
		Agrega espacio adicional alrededor del bounding box original en ambas direcciones
		(izquierda-derecha y arriba-abajo), asegurando que no se exceda el tamaño de la imagen.
		
		Args:
		x1, y1, x2, y2: Coordenadas originales del bounding box.
		img_shape: Forma de la imagen (altura, ancho, canales).
		margin: Factor de expansión como fracción del ancho/alto (default: 0.10 = 10%).
		
		Returns:
		Tupla (x1_new, y1_new, x2_new, y2_new) con el bounding box expandido y ajustado.
		"""
		
		h, w = img_shape[:2]

		bw = x2 - x1
		bh = y2 - y1

		extra_w = int(bw * self.config.margin / 2)
		extra_h = int(bh * self.config.margin / 2)

		x1 = max(0, x1 - extra_w)
		x2 = min(w, x2 + extra_w)
		y1 = max(0, y1 - extra_h)
		y2 = min(h, y2 + extra_h)

		return x1, y1, x2, y2

	def _recortar(
		self,
		img: ImgColor,
		mask: Mask,
		) -> tuple[ImgColor, Mask]:
		"""
		Recorta la imagen según la máscara y la redimensiona a un cuadrado de `self.config.target_size` x `self.config.target_size`.
		
		Pipeline:
		1. Extrae bounding box de la máscara.
		2. Expande el bbox con margen de 10% (o el elegido).
		3. Recorta imagen y máscara.
		4. Redimensiona manteniendo aspecto original.
		5. Centra el resultado en un lienzo cuadrado con relleno negro.
		
		Args:
		img: Imagen de entrada (BGR o escala de grises).
		mask: Máscara binaria asociada a la imagen.
		self.config.target_size: Dimensión final del cuadrado (default: 256).
		
		Returns:
		Tupla (img_sq, mask_sq) con imagen y máscara redimensionadas y centradas
		en un lienzo cuadrado de tamaño `self.config.target_size` x `self.config.target_size`.
		"""

		x1, y1, x2, y2 = self._bbox_desde_mask(mask)
		x1, y1, x2, y2 = self._expandir_bbox(x1, y1, x2, y2, img.shape)

		img_crop = img[y1:y2, x1:x2]
		mask_crop = mask[y1:y2, x1:x2]

		h, w = img_crop.shape[:2]
		scale = self.config.target_size / max(h, w)
		new_w = int(round(w * scale))
		new_h = int(round(h * scale))

		img_resized = cv.resize(img_crop, (new_w, new_h), interpolation=cv.INTER_AREA)
		mask_resized = cv.resize(mask_crop, (new_w, new_h), interpolation=cv.INTER_NEAREST)

		img_sq = np.zeros((self.config.target_size, self.config.target_size, 3), dtype=img.dtype)
		mask_sq = np.zeros((self.config.target_size, self.config.target_size), dtype=mask.dtype)

		y_off = (self.config.target_size - new_h) // 2
		x_off = (self.config.target_size - new_w) // 2

		img_sq[y_off:y_off+new_h, x_off:x_off+new_w] = img_resized
		mask_sq[y_off:y_off+new_h, x_off:x_off+new_w] = mask_resized

		return img_sq, mask_sq

	def _refinar_mask(
		self, 
		mask_sq: Mask,
		open_ksize: int = 3,
		close_ksize: int = 3
		) -> Mask:
		"""
		Refina la máscara aplicando operaciones morfológicas de apertura y cierre.
		Elimina ruido y pequeños artefactos en la máscara usando:
		1. Apertura (erosión + dilatación): remueve pequeños componentes.
		2. Cierre (dilatación + erosión): rellena pequeños huecos.
		
		Args:
		mask_sq: Máscara binaria de entrada (uint8).
		open_ksize: Tamaño del kernel para apertura (default: 3). Debe ser impar.
		close_ksize: Tamaño del kernel para cierre (default: 3). Debe ser impar.
		
		Returns:
		Máscara refinada (uint8) tras aplicar operaciones morfológicas.
		
		ValueError: 
		Si open_ksize o close_ksize no son números impares y/o menores a cero.
		"""
		open_ksize = self.config.open_ksize
		close_ksize = self.config.close_ksize

		if open_ksize <= 0 or close_ksize <= 0 or open_ksize % 2 == 0 or close_ksize % 2 == 0:
			raise ValueError("Los kernels deben ser impares y > 0.")

		kernel_open = np.ones((open_ksize, open_ksize), np.uint8)
		kernel_close = np.ones((close_ksize, close_ksize), np.uint8)

		mask_clean = cv.morphologyEx(mask_sq, cv.MORPH_OPEN, kernel_open, iterations=1)
		mask_clean = cv.morphologyEx(mask_clean, cv.MORPH_CLOSE, kernel_close, iterations=1)
		
		return mask_clean
