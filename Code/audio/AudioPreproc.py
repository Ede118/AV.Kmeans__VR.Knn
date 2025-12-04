from pathlib import Path
import numpy as np

import librosa
import librosa.effects as fx

import scipy.signal as sps
from scipy.signal import butter, sosfiltfilt
from scipy.io import wavfile

from math import gcd

from dataclasses import dataclass, field
from typing import Literal
from Code.types import VecF, F32, AudioSignal



# -------------------------------------------------------------------------------------------------  #
#                                    --------- Config  ---------                                     #
# -------------------------------------------------------------------------------------------------  #

@dataclass(frozen=True)
class AudioPreprocConfig:
	# Normalización de sampling y duración
	target_sr: int = 16000								# f_ss [Hz]
	t_sec: float = 1.2									# T_fija [s]

	# Ventaneo (para VAD y, luego, AudioFeat)
	frame_ms: float = 25.0
	hop_ms: float = 10.0

	# Filtro
	corte_pasaalto: float = 80.0        				# f_corte del pasa-alto
	orden_pasaalto: int = 4

	# Pre-énfasis (estándar: realza altas frecuencias, mejora SNR de formantes)
	coeficiente_pre_enfasis: float = 0.97				# y[n] = x[n] - a x[n-1]

	# Normalización de nivel
	norm_mode: Literal["RMS", "PEAK"] = "RMS"           # "rms" o "peak"
	rms_target_dbfs: float = -20.0   					# objetivo si mode="rms"
	peak_ref: float = 0.98           					# objetivo si mode="peak"
	max_gain_db: float = 18.0        					# límite de ganancia
	gate_dbfs: float = -60.0         					# no subir por debajo de este nivel

	# Relleno al fijar duración
	pad_mode: str = "edge"           					# "edge" | "constant" | "reflect"
	# Valor "mínimo" para tolerancias en cálculos numéricos
	epsilon: ClassVar[float] = float(np.finfo(np.float32).eps)

# -------------------------------------------------------------------------------------------------  #
#                                    --------- Clase  ---------                                      #
# -------------------------------------------------------------------------------------------------  #

@dataclass(slots=True)
class AudioPreproc:
	config: AudioPreprocConfig = field(default_factory=AudioPreprocConfig)

	# -------------------------------------------------------------------------------------------------  #
	#                           ---------- API pública ----------                                        #
	# -------------------------------------------------------------------------------------------------  #
	
	def procesar(
			self, 
			audio_path: str | Path, 
		) -> tuple[AudioSignal, int]:
		"""
		### Preprocesamiento completo
		Resamplea, filtra, realza, aplica VAD, normaliza y ajusta la duración.
		- Entrada: señal mono (`np.ndarray`) y sample rate original
		- Salida: audio limpio (`float32`) y sample rate objetivo (`config.target_sr`)
		### Resumen
		```
		audio_proc, sr_out = preproc.preprocesar(y, sr_in)
		```
		"""
		
		audio_path = Path(audio_path)
		if not audio_path.is_file():
			raise ValueError("No se ha pasado un path válido al archivo de audio.")


		# 1) Resample a target_sr y forzamos a mono
		# Además, librosa carga en float32 y normaliza a aprox [-1, 1]
		# Pero no asegura que PEAK = 1.0
		y, sampling_rate = librosa.load(
			path=str(audio_path),
			sr=self.config.target_sr,
			mono=True
		)

		# 2) Filtro pasa-alto (rumble fuera)
		y = self._filtro_pasa_alto(
			audio=y,
			sampling_rate=self.config.target_sr,
			frecuencia_corte=self.config.corte_pasaalto,
			order=self.config.orden_pasaalto
		)

		# 3) Pre-énfasis (realce de altas; NO compensa graves recortados)
		y = self._pre_enfasis(
			audio=y,
			coeficiente_pre_enfasis=self.config.coeficiente_pre_enfasis
		)

		# 4) VAD simple: recorta silencios y expande bordes
		y = self._simple_vad(
			audio=y,
			sampling_rate=self.config.target_sr,
			frame_ms=self.config.vad_frame_ms,
			hop_ms=self.config.vad_hop_ms,
			top_db=self.config.vad_top_db
		)

		# 5) Normalización de nivel (RMS o pico) con guardarraíles
		y = self._normalizar(
			audio=y,
			mode=self.config.norm_mode,
			RMS_dBFS=self.config.rms_target_dbfs,
			PEAK_ref=self.config.peak_ref,
			max_gain_dB=self.config.max_gain_db,
			gate_dBFS=self.config.gate_dbfs
		)

		# 6) Duración fija T_fija con padding elegido
		y = self._arreglar_duracion(
			y, 
			self.config.target_sr, 
			self.config.t_sec, 
			pad_mode=self.config.pad_mode, 
			center_crop=False
		)

		return y.astype(F32, copy=False), self.config.target_sr


	def parametros_de_framing(self) -> tuple[int, int]:
		"""
		### Ventana y hop
		Convierte `frame_ms` y `hop_ms` a muestras según `config.target_sr`.
		- Devuelve `(win_muestras, hop_muestras)`
		### Resumen
		```
		win, hop = preproc.framing_params()
		```
		"""
		sr = int(self.config.target_sr)
		win = max(1, int(round(sr * self.config.frame_ms / 1000.0)))
		hop = max(1, int(round(sr * self.config.hop_ms  / 1000.0)))
		return win, hop

	# -------------------------------------------------------------------------------------------------  #
	#                       ---------- Helpers Privados ----------                                       #
	# -------------------------------------------------------------------------------------------------  #

	def _filtro_pasa_alto(
		audio: AudioSignal,
		sampling_rate: int,
		frecuencia_corte: float = 80.0,
		order: int = 4,
		) -> AudioSignal:
		"""
		Filtro pasa alto Butterworth aplicado con fase cero.
		y: señal mono (1D)
		sr: sample rate de la señal
		cutoff_hz: frecuencia de corte (Hz)
		order: orden del filtro (2, 4, 6...)
		"""
		y = np.asarray(audio, dtype=F32).squeeze()
		if y.ndim != 1:
			raise ValueError("Se espera audio mono (vector 1D)")

		nyq = sampling_rate / 2.0
		norm_cutoff = frecuencia_corte / nyq
		if not 0 < norm_cutoff < 1:
			raise ValueError(f"frecuencia de corte={cutoff_hz} no tiene sentido para sampling rate={sampling_rate}")

		sos = butter(order, norm_cutoff, btype="highpass", output="sos")
		y_hp = sosfiltfilt(sos, y)

		return y_hp.astype(np.float32, copy=False)

	# -------------------------------------------------------------------------------------------------  #

	@staticmethod
	def _pre_enfasis(
		audio: AudioSignal,
		coeficiente_pre_enfasis: float = 0.97
		) -> AudioSignal:
		"""
		### Pre-énfasis
		Aplica `y[n] = x[n] - a·x[n-1]` para realzar altas frecuencias.
		- Conserva la nota original sobre el realce de altas frecuencias
		### Resumen
		```
		y_pe = AudioPreproc._pre_enfasis(y, 0.97)
		```
		"""
		y = np.asarray(audio, dtype=F32)
		alpha = float(coeficiente_pre_enfasis)
		
		if y.size == 0:
			return y

		if y.ndim != 1:
			raise ValueError("Se espera audio mono 1D")

		x1 = np.array([1.0, -alpha], dtype=np.float32)
		x2 = np.array([1.0], dtype=np.float32)

		y_filt = lfilter(x1, x2, y)
		return y_filt.astype(np.float32, copy=False)

	# -------------------------------------------------------------------------------------------------  #
	
	@staticmethod
	def _simple_vad(
		audio: AudioSignal,
		sampling_rate: int,
		frame_ms: float,
		hop_ms: float,
		top_db: float,
		) -> AudioSignal:
		"""
		Recorte de silencios usando librosa.effects.trim.

		- Recorta silencios al inicio y al final
		- Usa energía en dB relativa al máximo (top_db)
		- Mantiene solo el "núcleo" del comando de voz
		"""
		y = np.asarray(audio, dtype=F32).squeeze()
		if y.ndim != 1:
			raise ValueError("Se espera audio mono 1D")

		sr = int(sampling_rate)
		vad_frame_ms = float(frame_ms)
		vad_hop_ms = float(hop_ms)
		vad_top_db = float(top_db)

		frame_length = int(round(sr * vad_frame_ms / 1000.0))
		hop_length   = int(round(sr * vad_hop_ms  / 1000.0))

		# Evitar valores ridículos
		frame_length = max(1, frame_length)
		hop_length   = max(1, hop_length)

		y_trim, _ = fx.trim(
			y,
			top_db=vad_top_db,
			frame_length=frame_length,
			hop_length=hop_length,
		)

		return y_trim.astype(F32, copy=False)

	# -------------------------------------------------------------------------------------------------  #

	def _normalizar(
		self,
		audio: AudioSignal,
		mode: str,
		PEAK_ref: float = 0.98,
		RMS_dBFS: float = -20.0,
		max_gain_dB: float = 18.0,
		gate_dBFS: float = -60.0
	) -> AudioSignal:
		"""
		### Normalización de nivel
		Ajusta el volumen por pico o RMS con límites de ganancia.
		- Respeta `gate_dbfs` para evitar subir ruidos muy bajos
		### Resumen
		```
		y_norm = AudioPreproc._normalizar(y, mode="rms", rms_target_dbfs=-20)
		```
		"""
		mode = mode.upper()
		if mode not in ("PEAK", "RMS"):
			raise ValueError("Valor de \'mode\' inválido. Valores posibles: 'PEAK' / 'RMS'")

		y = audio.astype(F32, copy=False)
		epsilon = self.config.epsilon

		if mode == "PEAK":
			p = float(np.max(np.abs(y))) if y.size else 0.0
			if p < epsilon:
				raise ValueError("Se ha intentado normalizar una señal de energía casi nula.")
			gain = PEAK_ref / (p + epsilon)

		elif mode == "RMS":
			rms = float(np.sqrt(np.mean(y * y)) + epsilon)
			curr_db = 20.0 * np.log10(rms)
			if curr_db < gate_dBFS:
				return y
			delta = RMS_dBFS - curr_db
			gain = 10.0 ** (delta / 20.0)

		max_gain = 10.0 ** (max_gain_dB / 20.0)
		
		# Valor mínimo: 1.0 / max_gain (no atenuar)
		# Valor máximo: max_gain
		gain = float(np.clip(gain, 1.0 / max_gain, max_gain))

		z = y * gain

		return np.clip(z, -1.0, 1.0, out=z).astype(F32, copy=False)

	# -------------------------------------------------------------------------------------------------  #

	@staticmethod
	def _arreglar_duracion(
		audio: AudioSignal,
		sampling_rate: int,
		T_sec: float,
		pad_mode: str,
		center_crop: bool = False
	) -> AudioSignal:
		"""
		### Duración fija
		Recorta o rellena la señal para alcanzar `t_sec`.
		- Soporta modos de padding (`edge`, `constant`, `reflect`)
		### Resumen
		```
		y_pad = AudioPreproc._arreglar_duracion(y, sr, 1.2, pad_mode="edge")
		```
		"""
		y = audio.astype(F32, copy=False)
		N = int(round(sampling_rate * T_sec))
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
