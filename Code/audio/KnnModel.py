import numpy as np
from dataclasses import dataclass, field
from .Standardizer import Standardizer
from typing import Literal, Optional, Sequence, List

from Code.types import VecF, MatF, F32


@dataclass(frozen=True, slots=True)
class KnnConfig:
  k: int = 5
  metric: Literal["cosine", "euclidean"] = "cosine"
  weighted: bool = True
  eps: float = 1e-6
  reject_max_dist: Optional[float] = None
  ratio_max: Optional[float] = None


@dataclass(slots=True)
class KnnModel:
  cfg: KnnConfig = field(default_factory=KnnConfig)

  # estado tras "fit"
  X: Optional[MatF] = None            # (N, D) z-scoreado
  y_idx: Optional[np.ndarray] = None  # (N,) int32
  labels: Optional[tuple[str, ...]] = None
  _row_norms: Optional[VecF] = None  # (N,) para métrica coseno

  # -------------------------------------------------------------------------------------------------  #
  #                              --------- Módulos Públicos  ---------                                 #
  # -------------------------------------------------------------------------------------------------  #

  def upload_batch(
    self,
    matParametros: np.ndarray,   # (N, D) YA estandarizado
    vecLabel: Sequence[str]      # len N
  ) -> int:
    """
    ### Cargar referencias
    Almacena embeddings estandarizados y sus etiquetas.
    - Precalcula normas si la métrica es coseno
    ### Resumen
    ```
    knn.upload_batch(X_std, etiquetas)
    ```
    """
    X = np.asarray(matParametros, dtype=F32)
    if X.ndim != 2:
        raise ValueError("matParametros debe tener shape (N, D)")
    N, D = X.shape
    if len(vecLabel) != N:
        raise ValueError("vecLabel debe tener longitud N")

    # mapear labels a enteros estables
    index_of = {}
    uniq: List[str] = []
    y_idx = np.empty(N, dtype=np.int32)
    for i, lab in enumerate(vecLabel):
        if lab not in index_of:
            index_of[lab] = len(uniq)
            uniq.append(lab)
        y_idx[i] = index_of[lab]

    # guardar estado
    self.X = X
    self.y_idx = y_idx
    self.labels = tuple(uniq)

    # precálculo de normas para métrica coseno
    if self.cfg.metric == "cosine":
        norms = np.linalg.norm(X, axis=1)
        norms = np.where(norms < self.cfg.eps, 1.0, norms)
        self._row_norms = norms.astype(F32)
    else:
        self._row_norms = None

    return N

  def predict(
        self, 
        x: np.ndarray, 
        exclude_idx: int | None = None
      ) -> str:
    """
    ### Predicción k-NN
    Asigna una etiqueta a un vector ya estandarizado.
    - Permite excluir un índice para validación LOO
    ### Resumen
    ```
    etiqueta = knn.predict(x_std)
    ```
    """
    if self.X is None or self.y_idx is None or self.labels is None:
        raise RuntimeError("Modelo no inicializado. Llama upload_batch() primero.")

    x = np.asarray(x, dtype=F32).reshape(-1)
    D = self.X.shape[1]
    if x.shape[0] != D:
        raise ValueError(f"Dimensión de x inválida. Esperado (D,) con D={D}.")

    # distancias a todo el train
    d = self._distances(x)   # (N,)

    # LOO opcional: no te elijas a vos mismo
    if exclude_idx is not None and 0 <= exclude_idx < d.size:
        d = d.copy()
        d[exclude_idx] = np.inf

    y_hat_idx = self._vote(d)
    return self.labels[y_hat_idx]


  # -------------------------------------------------------------------------------------------------  #
  #                              --------- Módulos Privados  ---------                                 #
  # -------------------------------------------------------------------------------------------------  #

  def _distances(
      self,
      vecAudio: VecF,
  ) -> VecF:
    """
    ### Distancias a la base
    Calcula distancias (euclidianas o coseno) contra todos los vectores de referencia.
    - Devuelve `np.ndarray` de tamaño `N`
    ### Resumen
    ```
    distancias = knn._distances(x_std)
    ```
    """
    matDB = self.X
    vecAudio: VecF = np.asarray(vecAudio, dtype=F32).reshape(-1)

    if vecAudio.shape[0] != matDB.shape[1]:
      raise ValueError("x debe tener dimensión (D,)")
    
    if self.cfg.metric == "euclidean":
      diff = matDB - vecAudio[None, :]
      vecDistancias = np.sqrt(np.sum(diff*diff, axis=1))

      return vecDistancias.astype(F32)

    elif self.cfg.metric == "coseno":
      vec_norm = F32(np.linalg.vector_norm(vecAudio))

      if vec_norm < self.cfg.eps:
        raise ValueError("Es probablemente silencio.")
      
      norms = self._row_norms
      if norms is None:
        norms = np.linalg.norm(matDB, axis=1)
        norms = np.where(norms < self.cfg.eps, 1.0, norms)

      dots = matDB @ vecAudio
      vecDistancias = 1.0 - dots / (norms * vec_norm + self.cfg.eps)

      return np.clip(vecDistancias, 0.0, 2.0).astype(F32)
    
    else:
      raise TypeError("Solo se tienen cáculo de distancias euclidianas o de cosenos.")
  
  def _vote(
        self, 
        d: np.ndarray
      ) -> int:
    """
    ### Votación ponderada
    Selecciona la clase ganadora con top‑k y pesos 1/(d+eps).
    - Desempata usando el vecino individual más cercano
    ### Resumen
    ```
    clase = knn._vote(distancias)
    ```
    """
    if self.X is None or self.y_idx is None or self.labels is None:
        raise RuntimeError("Modelo no inicializado. Llama upload_batch() primero.")

    k = int(self.cfg.k)
    if k <= 0 or k > d.size:
        raise ValueError("k inválido para el tamaño del batch.")

    # top-k índices y ordenados por distancia ascendente
    idx_k = np.argpartition(d, k)[:k]
    order = np.argsort(d[idx_k])
    idx_k = idx_k[order]

    y_k = self.y_idx[idx_k]             # (k,)
    d_k = d[idx_k].astype(F32)   # (k,)

    # pesos
    eps = float(self.cfg.eps)
    d_k = np.maximum(d_k, eps)                    # evita 1/0
    if self.cfg.weighted:
        w = 1.0 / (d_k + eps)
    else:
        w = np.ones_like(d_k, dtype=F32)

    # voto ponderado por clase
    C = len(self.labels)
    scores = np.bincount(y_k, weights=w, minlength=C)  # (C,)

    # ganador + desempate por vecino más cercano
    winners = np.flatnonzero(scores == scores.max())
    if winners.size == 1:
        return int(winners[0])

    # empate: gana la clase cuyo vecino más cercano tenga menor d
    best_c = None
    best_d = np.inf
    for c in winners:
        d_min_c = float(d_k[y_k == c].min())
        if d_min_c < best_d:
            best_d = d_min_c
            best_c = int(c)
    return best_c

  # -------------------------------------------------------------------------------------------------  #
  #                                  --------- Alias públicos ---------                               #
  # -------------------------------------------------------------------------------------------------  #

  def cargar_referencias(
      self,
      matParametros: np.ndarray,
      vecLabel: Sequence[str]
  ) -> int:
    """
    ### Alias en español
    Delegación a `upload_batch`.
    ### Resumen
    ```
    knn.cargar_referencias(X_std, etiquetas)
    ```
    """
    return self.upload_batch(matParametros, vecLabel)

  def predecir(
      self,
      x: np.ndarray,
      exclude_idx: int | None = None
  ) -> str:
    """
    ### Alias en español
    Delegación a `predict`.
    ### Resumen
    ```
    etiqueta = knn.predecir(x_std)
    ```
    """
    return self.predict(x, exclude_idx=exclude_idx)

  def distancias(
      self,
      vecAudio: VecF,
  ) -> VecF:
    """
    ### Alias en español
    Delegación a `_distances`.
    ### Resumen
    ```
    dist = knn.distancias(x_std)
    ```
    """
    if self.X is None or self.y_idx is None or self.labels is None:
      raise RuntimeError("Modelo no inicializado. Llama upload_batch() primero.")
    return self._distances(vecAudio)
