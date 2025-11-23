from dataclasses import dataclass, field
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from typing import Tuple, List
import scipy.signal as sps

from Code.types import VecF, VecI, MatF, F32, I32, I8
from Code.audio.AudioPreproc import AudioPreproc, PreprocCfg as AudioPreprocCfg

try:
  import librosa
except Exception:
    raise RuntimeError("Esta función requiere 'librosa'. Instálalo en tu .venv.")


  # -------------------------------------------------------------------------------------------------  #
  #                              --------- Módulos Públicos  ---------                                 #
  # -------------------------------------------------------------------------------------------------  #

def _next_pow2(n: int) -> int:
  n = max(1, int(n))
  p = 1
  while p < n:
    p <<= 1
  return p

  # -------------------------------------------------------------------------------------------------  #

def _pool_stats(
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

def _delta_feat(
    mat: MatF,
    width: int = 9,
    order: int = 1
) -> MatF:
  
  if order == 0:
    raise ValueError("Dimensión debe ser entero positivo.")
  
  if width < 3:
        width = 3
  if width % 2 == 0:
        width += 1
  
  half = width // 2
  
  t = np.arange(-half, half + 1, dtype=F32)
  denom = np.sum(t**2)
  matPadding = np.pad(mat, ((0, 0), (half, half)), mode="edge")
  
  W = sliding_window_view(matPadding, window_shape=width, axis=1)

  d1 = (W * t).sum(axis=1) / denom
  d1 = d1.astype(F32, copy=False)

  if order == 1:
    return d1

  return _delta_feat(d1, width=width, order=order-1)

  # -------------------------------------------------------------------------------------------------  #

def _Hz2mel(
      f: np.ndarray
) -> np.ndarray:
  return 2595.0 * np.log10(1.0 + f / 700.0)

  # -------------------------------------------------------------------------------------------------  #

def _mel2Hz(
      m: np.ndarray
) -> np.ndarray:
  return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

  # -------------------------------------------------------------------------------------------------  #

def _mel_filterbank(
      sr: int, 
      n_fft: int, 
      n_mels: int = 40, 
      fmin: float = 0.0, 
      fmax: float = None
      ) -> MatF:
    """
    Banco de filtros mel triangular (n_mels, 1 + n_fft//2).
    """
    if fmax is None:
        fmax = sr / 2.0

    mels = np.linspace(_Hz2mel(fmin), _hz_to_hz := _Hz2mel(fmax), n_mels + 2)
    
    freqs = _mel2Hz(mels)
    
    # bins de FFT
    bins = np.floor((n_fft + 1) * freqs / sr).astype(int)
    fb = np.zeros((n_mels, 1 + n_fft // 2), dtype=F32)
    
    for m in range(1, n_mels + 1):
        f_m_minus, f_m, f_m_plus = bins[m - 1], bins[m], bins[m + 1]
        if f_m == f_m_minus: f_m += 1
        if f_m_plus == f_m:  f_m_plus += 1
        # subida
        fb[m - 1, f_m_minus:f_m] = np.linspace(0, 1, f_m - f_m_minus, endpoint=False, dtype=np.float32)
        # bajada
        fb[m - 1, f_m:f_m_plus] = np.linspace(1, 0, f_m_plus - f_m, endpoint=False, dtype=np.float32)
    
    # normalización tipo Slaney
    enorm = 2.0 / (freqs[2:n_mels+2] - freqs[:n_mels])
    fb *= enorm[:, None].astype(F32)
    return fb

  # -------------------------------------------------------------------------------------------------  #


def _dct_type_ii(
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

def _mfcc_fallback(
      y: MatF, 
      sr: int, 
      win: int, 
      hop: int, 
      n_mfcc_no_c0: int
      ) -> MatF:
    """
    MFCC sin librosa:
      - STFT (hann), |X|^2
      - banco mel (40)
      - log
      - DCT-II -> tomamos (n_mfcc_no_c0 + 1) y dropeamos c0
    Retorna (C, T) con C = n_mfcc_no_c0
    """
    f, t, Z = sps.stft(
        y.astype(F32),
        fs=sr,
        window="hann",
        nperseg=win,
        noverlap=win - hop,
        nfft=_next_pow2(win),
        boundary=None,
        padded=False,
        detrend=False,
        return_onesided=True
    )
    S = np.abs(Z).astype(F32)**2  # (freq_bins, T)
    n_mels = 40
    fb = _mel_filterbank(sr, _next_pow2(win), n_mels=n_mels, fmin=20.0, fmax=sr/2.0)  # (n_mels, freq_bins)
    M = np.maximum(fb @ S, 1e-12)  # (n_mels, T)
    logM = np.log(M)
    # necesitamos n_mfcc_no_c0 + 1 para poder dropear c0
    Cfull = _dct_type_ii(logM, n_out=n_mfcc_no_c0 + 1).astype(F32)  # (n_mfcc_no_c0+1, T)
    # dropear c0
    return Cfull[1:, :]  # (n_mfcc_no_c0, T)

  # -------------------------------------------------------------------------------------------------  #


def _mfcc_librosa(
      y: MatF, 
      sr: int, 
      win: int, 
      hop: int, 
      n_mfcc_no_c0: int
      ) -> np.ndarray:
    # pedimos uno más para luego dropear c0
    M = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc_no_c0 + 1,
        n_fft=_next_pow2(win),
        hop_length=hop,
        win_length=win,
        window="hann",
        center=False,
        htk=True,          # mel tipo HTK, consistente con muchos pipelines
        norm="ortho"       # hace la DCT ortonormal
    ).astype(F32)   # (n_mfcc_no_c0+1, T)
    return M[1:, :]        # (n_mfcc_no_c0, T)  sin c0

  # -------------------------------------------------------------------------------------------------  #


def _frame_starts(
      N: int, 
      win: int, 
      hop: int
      ) -> np.ndarray:
    last = N - win
    if last < 0:
        return np.array([], dtype=np.int64)
    return np.arange(0, last + 1, hop, dtype=np.int64)

def _rms_per_frame(
      y: np.ndarray, 
      win: int, 
      hop: int
      ) -> np.ndarray:
    N = len(y)
    idx = _frame_starts(N, win, hop)
    T = len(idx)
    rms = np.empty(T, dtype=F32)
    for i, s in enumerate(idx):
        seg = y[s:s+win]
        rms[i] = np.sqrt(np.mean(seg*seg, dtype=np.float64) + 1e-12)
    return rms[None, :]  # (1, T)

def _zcr_per_frame(
      y: np.ndarray, 
      win: int, 
      hop: int
      ) -> np.ndarray:
    N = len(y)
    idx = _frame_starts(N, win, hop)
    T = len(idx)
    z = np.empty(T, dtype=F32)
    for i, s in enumerate(idx):
        seg = y[s:s+win]
        # cruces de signo
        z[i] = np.count_nonzero(np.diff(np.signbit(seg))).astype(F32) / (win - 1 + 1e-9)
    return z[None, :]  # (1, T)

  # -------------------------------------------------------------------------------------------------  #
  #                                    --------- Clase  ---------                                      #
  # -------------------------------------------------------------------------------------------------  #


@dataclass(frozen=True)
class AudioFeatConfig:
  n_mfcc: int = 20              # coeficientes útiles (sin c0)
  delta_order: int = 1          # 0: sin d; 1: d; 2: dd
  add_rms: bool = True
  add_zcr: bool = True
  stats: Tuple[str, ...] = ("mean", "std", "p10", "p90")

@dataclass(slots=True)
class AudioFeat:
  """
  Extrae un vector fijo de features a partir de audio ya preprocesado.
  """
  # Inyección por defecto: crea un AudioPreproc con su config por defecto
  pre: AudioPreproc = field(default_factory=lambda: AudioPreproc(AudioPreprocCfg()))
  cfg: AudioFeatConfig = field(default_factory=AudioFeatConfig)

  # -------------------------------------------------------------------------------------------------  #

  def _extract_mfcc(
        self, 
        y: VecF, 
        sr: int, 
        win: int, 
        hop: int
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
    n = int(self.cfg.n_mfcc)
    

    if librosa is not None:
        M = _mfcc_librosa(y, sr, win, hop, n)   # (n, T) sin c0
    else:
        M = _mfcc_fallback(y, sr, win, hop, n)  # (n, T) sin c0

    feats: list[np.ndarray] = [M]

    # Derivadas si corresponde (recursivo dentro de _delta_feat)
    if self.cfg.delta_order >= 1:
        d1 = _delta_feat(M, width=9, order=1)
        feats.append(d1)
        if self.cfg.delta_order >= 2:
            d2 = _delta_feat(d1, width=9, order=1)  # Δ sobre Δ
            feats.append(d2)

    print([f.shape for f in feats])

    return np.concatenate(feats, axis=1).astype(np.float32)  # (C, T)

  # -------------------------------------------------------------------------------------------------  #

  def extract(
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
    
    win, hop = self.pre.parametros_de_framing()

    # 1) MFCC (sin c0) + delta/delta2 opcional
    MF = self._extract_mfcc(y, sr, win, hop)  # (Cmf, T)
    # Cmf, T = MF.shape  # (útil para asserts o logs)

    # 2) RMS y ZCR alineados a win/hop
    parts: list[np.ndarray] = [MF]
    if self.cfg.add_rms:
        parts.append(_rms_per_frame(y, win, hop))  # (1, T')
    if self.cfg.add_zcr:
        parts.append(_zcr_per_frame(y, win, hop))  # (1, T'')

    # Alinear tiempos por posibles off-by-one entre STFT y framing directo
    T_min = min(p.shape[1] for p in parts)
    if T_min == 0:
        raise ValueError("No se pudieron formar cuadros con los parámetros actuales (duración insuficiente).")
    parts = [p[:, :T_min] for p in parts]
    feat_mat = np.concatenate(parts, axis=0)  

    # 3) Pooling temporal a vector fijo
    vec = _pool_stats(feat_mat, self.cfg.stats)
    # Se retorna en float64 para asegurar precisión previa a la estandarización.
    return np.asarray(vec, dtype=np.float64)

  def _row_feature_names(self) -> List[str]:
    nombres: List[str] = []
    n = int(self.cfg.n_mfcc)
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

  def nombres_de_caracteristicas(
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
    filas = self._row_feature_names()
    etiquetas: List[str] = []
    for stat in stats_to_use:
      for nombre_base in filas:
        etiquetas.append(f"{nombre_base}_{stat}")
    return etiquetas
