
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple, List, Optional
import numpy as np
import scipy.signal as sps
from scipy.io import wavfile

from Code.types import VecF, MatF, I64, F64, F32, AudioSignal, Spectrogram



# -------------------------------------------------------------------------------------------------  #
#                                    --------- Config  ---------                                     #
# -------------------------------------------------------------------------------------------------  #

@dataclass(frozen=True)
class PreprocCfg:
    # Normalización de sampling y duración
    target_sr: int = 16_000          # f_ss [Hz]
    t_sec: float = 1.2               # T_fija [s]

    # Ventaneo (para VAD y, luego, AudioFeat)
    frame_ms: float = 25.0
    hop_ms: float = 10.0

    # Filtro
    highpass_hz: float = 40.0        # f_corte del pasa-alto
    hp_order: int = 2

    # Pre-énfasis (estándar: realza altas frecuencias, mejora SNR de formantes)
    preemph_a: float = 0.97          # y[n] = x[n] - a x[n-1]

    # VAD simple (basado en RMS en dBFS con post-proceso)
    vad_thresh_db: float = -35.0     # umbral de energía
    vad_win_ms: float = 20.0
    vad_min_ms: float = 120.0        # mínimo por segmento
    vad_expand_ms: float = 60.0      # expansión a ambos lados

    # Normalización de nivel
    norm_mode: str = "rms"           # "rms" o "peak"
    rms_target_dbfs: float = -20.0   # objetivo si mode="rms"
    peak_ref: float = 0.98           # objetivo si mode="peak"
    max_gain_db: float = 18.0        # límite de ganancia
    gate_dbfs: float = -60.0         # no subir por debajo de este nivel

    # Relleno al fijar duración
    pad_mode: str = "edge"           # "edge" | "constant" | "reflect"


# -------------------------------------------------------------------------------------------------  #
#                                    --------- Clase  ---------                                      #
# -------------------------------------------------------------------------------------------------  #

class AudioPreproc:
    def __init__(self, cfg: PreprocCfg = PreprocCfg()):
        self.cfg = cfg

    # -------------------------------------------------------------------------------------------------  #
    #                           ---------- API pública ----------                                        #
    # -------------------------------------------------------------------------------------------------  #
    
    def preprocesar(
            self, 
            y: AudioSignal, 
            sr: int
        ) -> Tuple[AudioSignal, int]:
        """
        ### Preprocesamiento completo
        Resamplea, filtra, realza, aplica VAD, normaliza y ajusta la duración.
        - Entrada: señal mono (`np.ndarray`) y sample rate original
        - Salida: audio limpio (`float32`) y sample rate objetivo (`cfg.target_sr`)
        ### Resumen
        ```
        audio_proc, sr_out = preproc.preprocess(y, sr_in)
        ```
        """
        
        y = np.asarray(y, dtype=F32).squeeze()
        if y.ndim != 1:
            raise ValueError("Se espera audio mono 1D; convierte a mono antes o usa _resample_mono")

        # 1) Resample a target_sr (y forzamos mono si corresponde)
        y = self._resample_mono(y, sr, self.cfg.target_sr)
        sr = self.cfg.target_sr

        # 2) Filtro pasa-alto (rumble fuera)
        y = self._highpass(y, sr, self.cfg.highpass_hz, self.cfg.hp_order)

        # 3) Pre-énfasis (realce de altas; NO compensa graves recortados)
        y = self._pre_enfasis(y, self.cfg.preemph_a)

        # 4) VAD simple: recorta silencios y expande bordes
        y = self._simple_vad(
            y, sr,
            thresh_db=self.cfg.vad_thresh_db,
            win_ms=self.cfg.vad_win_ms,
            min_ms=self.cfg.vad_min_ms,
            expand_ms=self.cfg.vad_expand_ms
        )

        # 5) Normalización de nivel (RMS o pico) con guardarraíles
        y = self._normalizar(
            y,
            mode=self.cfg.norm_mode,
            rms_target_dbfs=self.cfg.rms_target_dbfs,
            peak_ref=self.cfg.peak_ref,
            max_gain_db=self.cfg.max_gain_db,
            gate_dbfs=self.cfg.gate_dbfs
        )

        # 6) Duración fija T_fija con padding elegido
        y = self._arreglar_duracion(y, sr, self.cfg.t_sec, pad_mode=self.cfg.pad_mode, center_crop=False)

        return y.astype(F32, copy=False), sr

    def preprocesar_desde_path(self, path: str | Path) -> Tuple[VecF, int]:
        """
        ### Preprocesar desde disco
        Lee un WAV y ejecuta el pipeline de preprocesamiento.
        - Retorna audio procesado y sample rate objetivo
        ### Resumen
        ```
        audio_proc, sr_out = preproc.process_path("voz.wav")
        ```
        """
        sr, y = self._cargar_wav(path)
        y_proc, sr_proc = self.preprocesar(y, sr)
        return y_proc, sr_proc

    def parametros_de_framing(self) -> Tuple[int, int]:
        """
        ### Ventana y hop
        Convierte `frame_ms` y `hop_ms` a muestras según `cfg.target_sr`.
        - Devuelve `(win_muestras, hop_muestras)`
        ### Resumen
        ```
        win, hop = preproc.framing_params()
        ```
        """
        sr = int(self.cfg.target_sr)
        win = max(1, int(round(sr * self.cfg.frame_ms / 1000.0)))
        hop = max(1, int(round(sr * self.cfg.hop_ms  / 1000.0)))
        return win, hop

    # -------------------------------------------------------------------------------------------------  #
    #                       ---------- Helpers Privados ----------                                       #
    # -------------------------------------------------------------------------------------------------  #

    @staticmethod
    def _resample_mono(
        y: VecF, 
        sr_in: int, 
        sr_out: int
        ) -> VecF:
        """
        ### Resampleo mono
        Ajusta la señal a `sr_out` con `resample_poly`.
        - Solo reprocesa si `sr_in != sr_out`
        ### Resumen
        ```
        y_rs = AudioPreproc._resample_mono(y, 44100, 16000)
        ```
        """
        if sr_in == sr_out:
            return y.astype(F32, copy=False)
        # rational approximation
        from math import gcd
        g = gcd(sr_in, sr_out)
        up, down = sr_out // g, sr_in // g
        z = sps.resample_poly(y.astype(F32, copy=False), up, down, padtype="constant")
        return z.astype(F32, copy=False)

    # -------------------------------------------------------------------------------------------------  #

    @staticmethod
    def _highpass(
        y: VecF, 
        sr: int, 
        f0: float, 
        order: int = 4
        ) -> VecF:
        """
        ### Filtro pasa-alto
        Elimina rumble con un Butterworth en forma SOS.
        - Controlado por `f0` y `order`
        ### Resumen
        ```
        y_hp = AudioPreproc._highpass(y, 16000, 40.0, order=2)
        ```
        """
        nyq = 0.5 * sr
        wc = max(1.0, f0) / nyq
        wc = min(wc, 0.999)
        sos = sps.butter(order, wc, btype="highpass", output="sos")
        return sps.sosfiltfilt(sos, y).astype(F32, copy=False)

    # -------------------------------------------------------------------------------------------------  #

    @staticmethod
    def _pre_enfasis(
        y: VecF, 
        a: float = 0.97
        ) -> VecF:
        """
        ### Pre-énfasis
        Aplica `y[n] = x[n] - a·x[n-1]` para realzar altas frecuencias.
        - Conserva la nota original sobre el realce de altas frecuencias
        ### Resumen
        ```
        y_pe = AudioPreproc._pre_emphasis(y, 0.97)
        ```
        """
        if y.size == 0:
            return y
        z = np.empty_like(y)
        z[0] = y[0]
        z[1:] = y[1:] - a * y[:-1]
        return z.astype(F32, copy=False)

    # -------------------------------------------------------------------------------------------------  #

    def _simple_vad(
        self,
        y: VecF,
        sr: int,
        thresh_db: float = -35.0,
        win_ms: float = 20.0,
        min_ms: float = 120.0,
        expand_ms: float = 60.0
    ) -> VecF:
        """
        ### VAD simple
        Detecta y recorta silencios con un umbral de energía RMS.
        - Aplica limpieza de segmentos cortos y expansión temporal
        ### Resumen
        ```
        y_vad = preproc._simple_vad(y, sr, thresh_db=-35, win_ms=20, min_ms=120, expand_ms=60)
        ```
        """
        win = max(1, int(round(sr * win_ms / 1000.0)))
        hop = max(1, int(round(sr * self.cfg.hop_ms / 1000.0)))  # usa hop global para coherencia
        N = y.size
        if N < win:
            return y

        # RMS por frame y dBFS
        starts = np.arange(0, N - win + 1, hop, dtype=np.int64)
        rms = np.empty(starts.size, dtype=F32)
        for i, s in enumerate(starts):
            seg = y[s:s+win]
            rms[i] = np.sqrt(np.mean(seg*seg, dtype=np.float64) + 1e-12)
        db = 20.0 * np.log10(rms + 1e-12)

        # máscara por frame
        mask_f = db >= float(thresh_db)

        # eliminar islas cortas y expandir en FRAMES
        min_frames = max(1, int(round((min_ms / 1000.0 * sr - win) / hop)) + 1)
        expand_frames = max(0, int(round(expand_ms / 1000.0 * sr / hop)))

        # run-length smoothing
        if mask_f.any():
            # quitar runs < min_frames
            i = 0
            while i < mask_f.size:
                if mask_f[i]:
                    j = i + 1
                    while j < mask_f.size and mask_f[j]:
                        j += 1
                    if j - i < min_frames:
                        mask_f[i:j] = False
                    i = j
                else:
                    i += 1
            # expansión (dilatación 1D)
            if expand_frames > 0 and mask_f.any():
                idx = np.flatnonzero(mask_f)
                m = mask_f.copy()
                for k in idx:
                    i0 = max(0, k - expand_frames)
                    i1 = min(mask_f.size, k + expand_frames + 1)
                    m[i0:i1] = True
                mask_f = m

        # proyección de máscara de frames a muestras y recorte
        mask = np.zeros(N, dtype=bool)
        for i, s in enumerate(starts):
            if mask_f[i]:
                mask[s:s+win] = True

        if not mask.any():
            return y  # no se detectó voz; devolvemos original

        # recorta al bounding box activo
        idx = np.flatnonzero(mask)
        lo, hi = int(idx[0]), int(idx[-1]) + 1
        z = y[lo:hi]
        return z.astype(F32, copy=False)

    # -------------------------------------------------------------------------------------------------  #

    @staticmethod
    def _normalizar(
        y: VecF,
        mode: str = "rms",
        peak_ref: float = 0.98,
        rms_target_dbfs: float = -20.0,
        max_gain_db: float = 18.0,
        gate_dbfs: float = -60.0
    ) -> VecF:
        """
        ### Normalización de nivel
        Ajusta el volumen por pico o RMS con límites de ganancia.
        - Respeta `gate_dbfs` para evitar subir ruidos muy bajos
        ### Resumen
        ```
        y_norm = AudioPreproc._normalize(y, mode="rms", rms_target_dbfs=-20)
        ```
        """
        y = y.astype(F32, copy=False)
        eps = 1e-12

        if mode == "peak":
            p = float(np.max(np.abs(y))) if y.size else 0.0
            if p < eps:
                return y
            gain = peak_ref / (p + eps)

        elif mode == "rms":
            rms = float(np.sqrt(np.mean(y * y)) + eps)
            curr_db = 20.0 * np.log10(rms)
            if curr_db < gate_dbfs:
                return y
            delta = rms_target_dbfs - curr_db
            gain = 10.0 ** (delta / 20.0)
        else:
            raise ValueError("mode debe ser 'peak' o 'rms'")

        max_gain = 10.0 ** (max_gain_db / 20.0)
        gain = float(np.clip(gain, 1.0 / max_gain, max_gain))

        z = y * gain
        return np.clip(z, -1.0, 1.0, out=z).astype(F32, copy=False)

    # -------------------------------------------------------------------------------------------------  #

    @staticmethod
    def _arreglar_duracion(
        y: VecF,
        sr: int,
        t_sec: float,
        pad_mode: str = "edge",
        center_crop: bool = False
    ) -> VecF:
        """
        ### Duración fija
        Recorta o rellena la señal para alcanzar `t_sec`.
        - Soporta modos de padding (`edge`, `constant`, `reflect`)
        ### Resumen
        ```
        y_pad = AudioPreproc._fix_duration(y, sr, 1.2, pad_mode="edge")
        ```
        """
        y = y.astype(F32, copy=False)
        N = int(round(sr * t_sec))
        if N <= 0:
            return np.zeros(0, dtype=F32)

        n = y.size
        if n >= N:
            if center_crop:
                start = max(0, (n - N) // 2)
                return y[start:start+N].astype(F32, copy=False)
            return y[:N].astype(F32, copy=False)

        # padding
        pad = N - n
        if pad_mode == "edge":
            z = np.pad(y, (0, pad), mode="edge")
        elif pad_mode == "reflect":
            try:
                z = np.pad(y, (0, pad), mode="reflect")
            except Exception:
                z = np.pad(y, (0, pad), mode="constant", constant_values=0.0)
        elif pad_mode == "constant":
            z = np.pad(y, (0, pad), mode="constant", constant_values=0.0)
        else:
            raise ValueError("pad_mode inválido")
        return z.astype(F32, copy=False)

    # -------------------------------------------------------------------------------------------------  #

    @staticmethod
    def _cargar_wav(path: str | Path) -> Tuple[int, VecF]:
        """Read a WAV file as float32 mono in [-1, 1]."""
        path_obj = Path(path)
        if not path_obj.is_file():
            raise FileNotFoundError(f"No existe el archivo de audio: {path_obj}")
        sr, data = wavfile.read(path_obj)
        if data.ndim > 1:
            data = data.mean(axis=1)
        data = AudioPreproc._2float32(data)
        return sr, data

    @staticmethod
    def _2float32(y: np.ndarray) -> VecF:
        """Convert integer or float arrays to float32 in [-1, 1]."""
        arr = np.asarray(y)
        if not np.issubdtype(arr.dtype, np.floating):
            info = np.iinfo(arr.dtype)
            scale = max(abs(info.min), abs(info.max))
            arr = arr.astype(np.float32) / float(scale if scale else 1)
        else:
            arr = arr.astype(np.float32)
        return np.clip(arr, -1.0, 1.0).astype(np.float32, copy=False)
