import numpy as np
import cv2
from typing import Literal


from Code.types import MaskU8, VecF, GrayImageF32

class ImgFeat(object):
    """
    ### Explicación
    Extractor de características geométricas/fotométricas para objetos segmentados.
    
    Usa la imagen normalizada y la máscara asociada (producidas por `ImgPreproc`)
    para construir vectores de features que alimentan modelos de clustering o clasificación.
    
    ### Métodos principales:
    
    - `shape_vector`: arma un descriptor compacto (3D o 5D) con huecos, circularidad,
      vértices, rugosidad y gradiente.
    - `contar_huecos`, `contar_vertices`, `circularidad`, `rugosidad`, `gradiente_interno`:
      calculan métricas atómicas reutilizables en otros pipelines.
    - `feature_names`: devuelve etiquetas legibles para cada componente del descriptor.
    
    Las entradas (`img_norm`, `mask`) deben estar alineadas y en escala 0..1 (imagen)
    y {0,1}/{0,255} (máscara). Cada helper robustifica la máscara para tolerar ambos rangos.
    """


    def shape_vector(self,
            img_norm: GrayImageF32,
            mask: MaskU8,
            mask_type: Literal['binary', 'zero_one'] = 'binary',
            dim: Literal['3D', '5D'] = '5D',
            usar_gradiente_en_3D: bool = True) -> VecF:
        """
        Dada una imágen con su máscara e imagen normalizada, se calcula
        el vector de parámetros para clasificación con algorítmos K means.

        La dimensión de este vector define la dimensión del espacio de trabajo
        del algoritmo K means

        La cantidad de parámetros considerados son 5:
        - cantidad de huecos            : `int`
        - circularidad C                : `float`
        - cantidad de vertices (aprox)  : `int`
        - rugosidad R                   : `float`
        - gradiente de intensidad G     : `float`

        El orden es como fue descrito anteriormente.

        Parámetros
        ----------
        `img_norm` : `np.ndarray (float32, 0..1)`
            Imagen normalizada (gris) salida de ImgPreproc.normalize().
        `mask` : `np.ndarray (uint8, {0,255} o {0,1})`
            Máscara binaria salida de ImgPreproc.segment().
        `dim` : {'3D','5D'}
            '5D' -> [huecos, circularidad, vertices, rugosidad, gradiente]
            '3D' -> [huecos, circularidad, X] con X = gradiente o rugosidad.
        `usar_gradiente_en_3D` : bool
            Si True, el tercer eje en 3D es gradiente; si False, rugosidad.

        Retorna
        -------
        `np.ndarray (float32)`
            Vector de dimensión 3 o 5 según `dim`.
        """

        # --- validaciones mínimas ---
        if dim not in {'3D', '5D'}:
            raise ValueError(f"dim debe ser '3D' o '5D', no '{dim}'")

        if img_norm is None or mask is None:
            raise ValueError("img_norm y mask no pueden ser None")

        # Asegurar tipos esperados
        img = img_norm.astype(np.float32, copy=False)
        m = mask

        # Binariza la máscara si es necesario
        if m.dtype != np.uint8 or (m.max() not in (1, 255)):
            # re-binariza por si viene en {0,1} o con grises
            if mask_type == 'binary':
                m = (m > 127).astype(np.uint8) * 255
            else:  # 'zero_one'
                m = (m > 0).astype(np.uint8) * 255
        if m.dtype == np.uint8 and m.max() == 1:
            m = (m > 0).astype(np.uint8) * 255


        # --- calcular features atómicas ---
        # Asumimos que estas funciones existen en este módulo/clase:
        # contar_huecos(mask) -> int
        # circularidad(mask) -> float
        # contar_vertices(mask, eps_ratio=0.02, usar_hull=True) -> int
        # rugosidad(mask) -> float
        # gradiente_interno(img_norm, mask, erosion_iters=1) -> float

        h = float(self.contar_huecos(m))
        c = float(self.circularidad(m))
        v = float(self.contar_vertices(m, eps_ratio=0.02, usar_hull=True))
        r = float(self.rugosidad(m))
        g = float(self.gradiente_interno(img, m, erosion_iters=1))

        if dim == '5D':
            n = np.zeros(5, dtype=np.float32)
            n[0] = h
            n[1] = c
            n[2] = v
            n[3] = r
            n[4] = g
            return n

        # dim == '3D'
        n = np.zeros(3, dtype=np.float32)
        n[0] = h
        n[1] = c
        n[2] = g if usar_gradiente_en_3D else r
        return n


    # -------------------------------------------------------------------------------------------------  #
    #                               --------- Funciones públicas  ---------                              #
    # -------------------------------------------------------------------------------------------------  #

    def contar_huecos(
        self, 
        mask: MaskU8
        ) -> int:
        """
        ### Conteo de huecos
        Calcula cuántos agujeros internos tiene el objeto segmentado.
        - Convierte la máscara a binaria {0,255}
        - Usa `cv2.findContours` con `RETR_CCOMP` para identificar hijos
        - Cuenta cuántos contornos tienen parent distinto de -1
        ### Resumen

        ```
        feats = ImgFeat()
        n_holes = feats.contar_huecos(mask)
        ```
        """


        m = (mask > 0).astype('uint8') * 255

        contours, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hier is None or len(contours) == 0:
            return 0

        # Con CCOMP, un 'parent != -1' indica agujero (hijo de un contorno externo)
        holes = int(np.sum(hier[0, :, 3] != -1))
        return holes

    # -------------------------------------------------------------------------------------------------  #

    def contar_vertices(
        self, 
        mask: MaskU8,
        eps_ratio: float = 0.02,
        usar_hull: bool = True
        ) -> int:
        """
        ### Conteo de vértices
        Aproxima el contorno del objeto y devuelve su número de vértices.
        - Normaliza la máscara a {0,255}
        - Obtiene el contorno externo principal
        - Opcionalmente usa casco convexo y `approxPolyDP` con tolerancia relativa
        ### Resumen

        ```
        feats = ImgFeat()
        n_vertices = feats.contar_vertices(mask, eps_ratio=0.02, usar_hull=True)
        ```
        """
        # 0) Asegurar binario uint8 {0,255}
        if mask.dtype != np.uint8 or (mask.max() not in (1, 255)):
            m = (mask > 0).astype(np.uint8) * 255
        else:
            # normalizar por si hay 1/0 en vez de 255/0
            m = (mask > 0).astype(np.uint8) * 255

        # 1) Contornos externos
        res = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contornos = res[0] if len(res) == 2 else res[1]
        if not contornos:
            return 0

        # 2) Contorno principal
        c = max(contornos, key=cv2.contourArea)
        if c is None or len(c) < 3:
            return 0

        # 3) Opcional: casco convexo
        if usar_hull:
            c = cv2.convexHull(c)

        # 4) Aproximación poligonal (tolerancia relativa al perímetro)
        per = cv2.arcLength(c, True)
        if per <= 0:
            return 0
        eps = max(1e-6, eps_ratio) * per
        aprox = cv2.approxPolyDP(c, eps, True)  # (L,1,2)

        return int(len(aprox))

    # -------------------------------------------------------------------------------------------------  #


    def circularidad(
        self, 
        mask: MaskU8
        ) -> float:
        """
        ### Circularidad
        Mide qué tan circular es el objeto principal (1.0 ≈ círculo perfecto).
        - Normaliza máscara y toma contorno externo mayor
        - Calcula área y perímetro en subpíxel
        - Retorna 4πA / P² (invariante a escala)
        ### Resumen

        ```
        feats = ImgFeat()
        circ = feats.circularidad(mask)
        ```
        """

        # 1) binario uint8 {0,255}
        # Transforma 128 x 128 de varios valores ->  128 x 128 de 0 - 255 (valores binarios)
        m = (mask > 0).astype('uint8') * 255

        # 2) contornos externos (ignora agujeros)
        cnts, _ = cv2.findContours(m.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return 0.0  # no hay objeto

        # 3) tomar el contorno más grande por área
        c = max(cnts, key=cv2.contourArea)

        # 4) medidas geométricas suaves (subpíxel)
        A = float(cv2.contourArea(c))                 # área del contorno externo
        P = float(cv2.arcLength(c, closed=True))      # perímetro
        if P <= 1e-9 or A <= 1e-9:
            return 0.0

        C = 4.0 * np.pi * A / (P * P)                 # invariante a escala
        return float(C)

    # -------------------------------------------------------------------------------------------------  #

    def rugosidad(
        self, 
        mask: MaskU8
        ) -> float:
        
        """
        ### Rugosidad
        Compara el perímetro real con el del casco convexo para cuantificar irregularidades.
        - Extrae contorno externo
        - Calcula perímetro original y del casco convexo
        - Retorna R = P / P_convexo - 1 (>= 0)
        ### Resumen

        ```
        feats = ImgFeat()
        rough = feats.rugosidad(mask)
        ```
        """


        m = (mask > 0).astype(np.uint8) * 255
        if cv2.countNonZero(m) == 0:
            return 0.0

        # 1) Contorno externo (1-pixel)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            return 0.0
        
        c = max(cnts, key=cv2.contourArea)

        # 2) Perímetros
        P = float(cv2.arcLength(c, True))
        hull = cv2.convexHull(c)
        P_h = float(cv2.arcLength(hull, True))
        if P_h <= 1e-9:
            return 0.0

        # 3) Rugosidad (adimensional, invariante a escala)
        R = P / P_h - 1.0
        return float(max(0.0, R))

    # -------------------------------------------------------------------------------------------------  #

    def gradiente_interno(
        self, 
        img_norm: GrayImageF32,
        mask: MaskU8,
        erosion_iters: int = 1
        ) -> float:
        """
        ### Gradiente interno
        Estima la energía de gradiente dentro del objeto evitando el borde.
        - Convierte la máscara y la erosiona `erosion_iters` veces
        - Calcula gradiente Sobel sobre la imagen normalizada
        - Devuelve la mediana de magnitudes dentro del interior erosionado
        ### Resumen

        ```
        feats = ImgFeat()
        grad = feats.gradiente_interno(img_norm, mask, erosion_iters=1)
        ```
        """
        # 0) sanity checks mínimos
        if img_norm is None or mask is None:
            return 0.0
        if img_norm.dtype != np.float32:
            img = img_norm.astype(np.float32)
        else:
            img = img_norm

        # 1) interior puro: erosión leve para sacar el borde externo
        m = (mask > 0).astype(np.uint8) * 255  # blindaje por si viene 0/1
        if erosion_iters > 0:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            m_in = cv2.erode(m, k, iterations=erosion_iters)
        else:
            m_in = m

        area = cv2.countNonZero(m_in)
        if area == 0:
            return 0.0  # objeto demasiado fino o máscara mala

        # 2) gradiente Sobel en 0..1
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)  # sqrt(gx^2 + gy^2)

        # 3) estadístico robusto en el interior erosionado
        vals = mag[m_in > 0]
        if vals.size == 0:
            return 0.0
        G = float(np.median(vals))
        return G
    
    # -------------------------------------------------------------------------------------------------  #

    def feature_names(
        self,
        dim: str = "5D",
        usar_gradiente_en_3D: bool = True
        ) -> list[str]:
        """
        ### Nombres de features
        Genera la lista de etiquetas para cada componente del vector de características.
        - Siempre incluye huecos y circularidad
        - Añade vértices, rugosidad y gradiente según la dimensión solicitada
        ### Resumen

        ```
        feats = ImgFeat()
        labels = feats.feature_names(dim="3D", usar_gradiente_en_3D=True)
        ```
        """

        base = ["huecos", "circularidad"]
        if dim == "5D":
            return base + ["vertices", "rugosidad", "gradiente"]
        # 3D
        return base + (["gradiente"] if usar_gradiente_en_3D else ["rugosidad"])

    def nombres_de_caracteristicas(
        self,
        dim: str = "5D",
        usar_gradiente_en_3D: bool = True
        ) -> list[str]:
        return self.feature_names(dim=dim, usar_gradiente_en_3D=usar_gradiente_en_3D)
