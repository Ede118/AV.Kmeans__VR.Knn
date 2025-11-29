from dataclasses import dataclass, field
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from typing import Tuple, List
import scipy.signal as sps
import signal

from Code.types import VecF, VecI, MatF, F32, I32, I8
from Code.audio.AudioPreproc import AudioPreproc, AudioPreprocConfig

try:
  import librosa
except Exception:
    raise RuntimeError("Esta función requiere 'librosa'. Instálalo en tu .venv.")


  # -------------------------------------------------------------------------------------------------  #
  #                              --------- Módulos Públicos  ---------                                 #
  # -------------------------------------------------------------------------------------------------  #

@dataclass(frozen=True)
class AudioFeatConfig:
  sr_target: float = 16e3
  win: float = 25e-3
  hop: float = 10e-3
  n_mfcc_no_c0: int = 20              # coeficientes útiles (sin c0)
  delta_order: int = 1          # 0: sin d; 1: d; 2: dd
  add_rms: bool = True
  add_zcr: bool = True
  stats: Tuple[str, ...] = ("mean", "std", "p10", "p90")


  # -------------------------------------------------------------------------------------------------  #
  #                                    --------- Clase  ---------                                      #
  # -------------------------------------------------------------------------------------------------  #

@dataclass(slots=True)
class AudioFeat:
  """
  Extrae un vector fijo de features a partir de audio ya preprocesado.
  """
  # Inyección por defecto: crea un AudioPreproc con su config por defecto
  cfg: AudioFeatConfig = field(default_factory=AudioFeatConfig)

  # -------------------------------------------------------------------------------------------------  #

  def extraer_caracteristicas(
        self, 
        y: VecF, 
        sr: int
    ) -> VecF:
    """
    ### Vector de características
    Combina MFCC, RMS y ZCR; aplica pooling y devuelve vector 1D `float64`.
    - Requiere audio generado por `AudioPreproc.preprocess`
    ### Resumen
    ```
    vec = feat.extract(y_proc, sr)
    ```
    """
    # Normalizamos tipo/shape: librosa exige np.ndarray 1D float
    y = np.asarray(y, dtype=np.float32).squeeze()
    if y.ndim != 1:
      raise ValueError("Se espera audio mono 1D; recibí forma {}".format(y.shape))

    # cfg.win/cfg.hop se expresan en segundos; convertimos a muestras
    win = max(1, int(self.cfg.win * sr))
    hop = max(1, int(self.cfg.hop * sr))

    # 1) MFCC (sin c0) + delta/delta2 opcional
    MF = self._extract_mfcc(y, sr, win, hop)  # (Cmf, T)
    # Cmf, T = MF.shape  # (útil para asserts o logs)

    # 2) RMS y ZCR alineados a win/hop
    parts: list[np.ndarray] = [MF]
    if self.cfg.add_rms:
        parts.append(self._rms_per_frame(y, win, hop))  # (1, T')
    if self.cfg.add_zcr:
        parts.append(self._zcr_per_frame(y, win, hop))  # (1, T'')

    # Alinear tiempos por posibles off-by-one entre STFT y framing directo
    T_min = min(p.shape[1] for p in parts)
    if T_min == 0:
        raise ValueError("No se pudieron formar cuadros con los parámetros actuales (duración insuficiente).")
    parts = [p[:, :T_min] for p in parts]
    feat_mat = np.concatenate(parts, axis=0)  

    # 3) Pooling temporal a vector fijo
    vec = self._calculo_estadisticos(feat_mat, self.cfg.stats)
    # Se retorna en float64 para asegurar precisión previa a la estandarización.
    return np.asarray(vec, dtype=np.float64)


  def nombres_features(
        self,
        stats: Tuple[str, ...] | None = None
    ) -> List[str]:
    """
    ### Etiquetas de features
    Devuelve nombres legibles alineados al vector de salida.
    - Respeta el orden exacto de `extract`
    ### Resumen
    ```
    nombres = feat.nombres_de_caracteristicas()
    ```
    """
    stats_to_use = stats if stats is not None else self.cfg.stats
    filas = self._nombre_canales()
    etiquetas: List[str] = []
    for stat in stats_to_use:
      for nombre_base in filas:
        etiquetas.append(f"{nombre_base}_{stat}")
    return etiquetas

    # -------------------------------------------------------------------------------------------------  #
    #                       ---------- Helpers Privados ----------                                       #
    # -------------------------------------------------------------------------------------------------  #
  
  def _nombre_canales(self) -> List[str]:
    nombres: List[str] = []
    n = int(self.cfg.n_mfcc_no_c0)
    for i in range(n):
      nombres.append(f"mfcc_{i+1}")
    if self.cfg.delta_order >= 1:
      for i in range(n):
        nombres.append(f"mfcc_delta_{i+1}")
      if self.cfg.delta_order >= 2:
        for i in range(n):
          nombres.append(f"mfcc_delta2_{i+1}")
    if self.cfg.add_rms:
      nombres.append("rms")
    if self.cfg.add_zcr:
      nombres.append("zcr")
    return nombres

    # -------------------------------------------------------------------------------------------------  #

  def _calculo_estadisticos(
      self,
      matInfo: MatF,
      stats: Tuple[str, ...]
    ) -> MatF:
    acc = []
    if "mean" in stats: acc.append(np.mean(matInfo, axis=1))
    if "std"  in stats: acc.append(np.std(matInfo, axis=1))
    if "p10"  in stats: acc.append(np.percentile(matInfo, 10, axis=1))
    if "p90"  in stats: acc.append(np.percentile(matInfo, 90, axis=1))
    return np.concatenate(acc, axis=0).astype(F32)

  # -------------------------------------------------------------------------------------------------  #

  # -------------------------------------------------------------------------------------------------  #

  def _Hz2mel(
    self,
    f: np.ndarray
    ) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + f / 700.0)

    # -------------------------------------------------------------------------------------------------  #

  def _mel2Hz(
    self,
      m: np.ndarray
    ) -> np.ndarray:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    # -------------------------------------------------------------------------------------------------  #

  def _dct_type_ii(
    self,
    x: MatF, 
    n_out: MatF
    ) -> MatF:
      """
      DCT-II por canal: entrada (n_mels, T) -> salida (n_out, T)
      """
      n_mels, T = x.shape
      # matriz DCT-II (sin normalización ortonormal porque es constante a escala)
      k = np.arange(n_out)[:, None]
      n = np.arange(n_mels)[None, :]
      dct = np.cos(np.pi / n_mels * (n + 0.5) * k).astype(np.float32)
      return (dct @ x).astype(np.float32)

    # -------------------------------------------------------------------------------------------------  #

  def _extract_mfcc(
        self, 
        y: VecF, 
        sr: int, 
        win: float, 
        hop: float
    ) -> VecF:
    """
    ### MFCC y derivadas
    Calcula MFCC sin c0 y añade Δ/ΔΔ según `cfg.delta_order`.
    - Devuelve matriz `(C, T)` en `float32`
    ### Resumen
    ```
    mat_mfcc = feat._extract_mfcc(y_proc, sr, win, hop)
    ```
    """
    n = int(self.cfg.n_mfcc_no_c0)
    # usar los parámetros que ya recibimos y el sr real de la señal
    y = np.asarray(y, dtype=np.float32).reshape(-1)

    n_ftt = int(2 ** np.ceil(np.log2(win)))

    M = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc= n + 1,
        n_fft=n_ftt,
        hop_length=hop,
        win_length=win,
        window="hann",
        center=False,
        htk=True,          # mel tipo HTK, consistente con muchos pipelines
        norm="ortho"       # hace la DCT ortonormal
    ).astype(F32)   # (n_mfcc_no_c0+1, T)
    
    M = M[1:, :]  
    
    feats: list[np.ndarray] = [M]

    # Derivadas si corresponde
    if self.cfg.delta_order >= 1:
      width = 9  
      d1 = librosa.feature.delta(M, width=width, order=1, axis=1, mode="nearest")
      feats.append(d1)

    if self.cfg.delta_order >= 2:
      d2 = librosa.feature.delta(d1, width=width, order=1, axis=1, mode="nearest")
      feats.append(d2)

    if self.cfg.delta_order >= 3:
      raise ValueError("Solo se aceptan hasta 2da derivada (delta).")

    return np.concatenate(feats, axis=0).astype(np.float32)  # (C, T)


  # -------------------------------------------------------------------------------------------------  #

  def _inicio_frames(
    self,
    N: int, 
    win: float, 
    hop: float
    ) -> np.ndarray:
      last = N - win
      if last < 0:
          return np.array([], dtype=np.int64)
      return np.arange(0, last + 1, hop, dtype=np.int64)

  def _rms_per_frame(self, y: np.ndarray, win: float, hop: float) -> np.ndarray:
      y = np.asarray(y, dtype=F32, order="C")
      if y.size < win:
          return np.empty((1, 0), dtype=F32)
      rms = librosa.feature.rms(
          y=y,
          frame_length=win,
          hop_length=hop,
          center=False
      ).astype(F32, copy=False)  # (1, T)
      return rms

  def _zcr_per_frame(self, y: np.ndarray, win: float, hop: float) -> np.ndarray:
      y = np.asarray(y, dtype=F32, order="C")
      if y.size < win:
          return np.empty((1, 0), dtype=F32)
      zcr = librosa.feature.zero_crossing_rate(
          y=y,
          frame_length=win,
          hop_length=hop,
          center=False
      ).astype(F32, copy=False)  # (1, T)
      return zcr
