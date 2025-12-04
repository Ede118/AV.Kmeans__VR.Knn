from dataclasses import dataclass, field
import numpy as np
import signal

from Code.types import VecF, MatF, F32, AudioSignal, FeatVec, FeatMat
from Code.audio.AudioPreproc import AudioPreproc, AudioPreprocConfig

import librosa



# -------------------------------------------------------------------------------------------------  #
#                              --------- Módulos Públicos  ---------                                 #
# -------------------------------------------------------------------------------------------------  #

@dataclass(frozen=True)
class AudioFeatConfig:
	sr_target: float = 16e3
	win_ms: float = 25.
	hop_ms: float = 10.
	N_MFCC: int = 20							# coeficientes útiles (sin c0)
	delta_order: int = 1						# 0: sin d; 1: d; 2: dd
	RMS: bool = True
	ZCR: bool = True
	stats: tuple[str, ...] = ("mean", "std", "p10", "p90")


	# -------------------------------------------------------------------------------------------------  #
	#                                    --------- Clase  ---------                                      #
	# -------------------------------------------------------------------------------------------------  #

@dataclass(slots=True)
class AudioFeat:
	"""
	Extrae un vector fijo de features a partir de audio ya preprocesado.
	"""
	# Inyección por defecto: crea un AudioPreproc con su config por defecto
	config: AudioFeatConfig = field(default_factory=AudioFeatConfig)

	# -------------------------------------------------------------------------------------------------  #

	def extraer_caracteristicas(
		self, 
		audio: AudioSignal, 
		sampling_rate: int
	) -> FeatVec:
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
		y = np.asarray(audio, dtype=np.float32).squeeze()
		if y.ndim != 1:
			raise ValueError("Se espera audio mono 1D; recibí forma {}".format(y.shape))

		# cfg.win/cfg.hop se expresan en segundos; convertimos a muestras
		win = max(1, int(self.config.win_ms * sampling_rate / 1000.0))
		hop = max(1, int(self.config.hop_ms * sampling_rate / 1000.0))

		# 1) MFCC (sin c0) + delta/delta2 opcional (derivadas)
		MFCC = self._calcular_MFCC_y_derivadas(
			audio=y, 
			sampling_rate=sampling_rate, 
			win_frames=win, 
			hop_frames=hop
		)  # (Cmf, T)

		# 2) RMS y ZCR alineados a win/hop
		parts: list[np.ndarray] = [MFCC]
		if self.config.RMS:
				parts.append(self._calcular_RMS(y, win, hop))  # (1, T')
		if self.config.ZCR:
				parts.append(self._calcular_ZCR(y, win, hop))  # (1, T'')

		# Alinear tiempos por posibles off-by-one entre STFT y framing directo
		T_min = min(p.shape[1] for p in parts)
		if T_min == 0:
				raise ValueError("No se pudieron formar cuadros con los parámetros actuales (duración insuficiente).")
		parts = [p[:, :T_min] for p in parts]
		feat_mat = np.concatenate(parts, axis=0)  

		# 3) Pooling temporal a vector fijo
		vec = self._calculo_estadisticos(feat_mat, self.config.stats)
		# Se retorna en float64 para asegurar precisión previa a la estandarización.
		return np.asarray(vec, dtype=np.float64)


	def nombres_features(
		self,
		stats: tuple[str, ...] | None = None
	) -> list[str]:
		"""
		### Etiquetas de features
		Devuelve nombres legibles alineados al vector de salida.
		- Respeta el orden exacto de `extract`
		### Resumen
		```
		nombres = feat.nombres_de_caracteristicas()
		```
		"""
		stats_to_use = stats if stats is not None else self.config.stats
		filas = self._nombre_canales()
		etiquetas: list[str] = []
		for stat in stats_to_use:
			for nombre_base in filas:
				etiquetas.append(f"{nombre_base}_{stat}")
		return etiquetas

	# -------------------------------------------------------------------------------------------------  #
	#                       ---------- Helpers Privados ----------                                       #
	# -------------------------------------------------------------------------------------------------  #
	
	def _nombre_canales(self) -> list[str]:
		nombres: list[str] = []
		n = int(self.config.N_MFCC)
		for i in range(n):
			nombres.append(f"MFCC {i+1}")
		if self.config.delta_order >= 1:
			for i in range(n):
				nombres.append(f"Δ {i+1}")
			if self.config.delta_order >= 2:
				for i in range(n):
					nombres.append(f"Δ² {i+1}")
		if self.config.RMS:
			nombres.append("RMS")
		if self.config.ZCR:
			nombres.append("ZCR")
		return nombres

	# -------------------------------------------------------------------------------------------------  #
	
	@staticmethod
	def _calculo_estadisticos(
		matInfo: MatF,
		stats: tuple[str, ...]
	) -> MatF:
		acc = []
		if "mean" in stats: acc.append(np.mean(matInfo, axis=1))
		if "std"  in stats: acc.append(np.std(matInfo, axis=1))
		if "p10"  in stats: acc.append(np.percentile(matInfo, 10, axis=1))
		if "p90"  in stats: acc.append(np.percentile(matInfo, 90, axis=1))
		return np.concatenate(acc, axis=0).astype(F32)

	# -------------------------------------------------------------------------------------------------  #

	# -------------------------------------------------------------------------------------------------  #

	@staticmethod
	def _Hz2mel(
		frecuencia_Hz: np.ndarray
	) -> np.ndarray:
		return 2595.0 * np.log10(1.0 + frecuencia_Hz / 700.0)

	# -------------------------------------------------------------------------------------------------  #

	@staticmethod
	def _mel2Hz(
		frecuencia_mel: np.ndarray
		) -> np.ndarray:
		return 700.0 * (10.0 ** (frecuencia_mel / 2595.0) - 1.0)

	# -------------------------------------------------------------------------------------------------  #

	@staticmethod
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

	def _calcular_MFCC_y_derivadas(
		self, 
		audio: AudioSignal, 
		sampling_rate: int, 
		win_frames: float, 
		hop_frames: float
	) -> FeatMat:
		"""
		### MFCC y derivadas
		Calcula MFCC sin c0 y añade Δ/ΔΔ según `cfg.delta_order`.
		- Devuelve matriz `(C, T)` en `float32`
		### Resumen
		```
		mat_mfcc = feat._extract_mfcc(y_proc, sr, win, hop)
		```
		"""
		n = int(self.config.N_MFCC)
		# usar los parámetros que ya recibimos y el sr real de la señal
		y = np.asarray(audio, dtype=np.float32).reshape(-1)

		n_ftt = int(2 ** np.ceil(np.log2(win_frames)))

		M = librosa.feature.mfcc(
				y=y,
				sr=sampling_rate,
				n_mfcc= n + 1,
				n_fft=n_ftt,
				hop_length=hop_frames,
				win_length=win_frames,
				window="hann",
				center=False,
				htk=True,          # mel tipo HTK, consistente con muchos pipelines
				norm="ortho"       # hace la DCT ortonormal
		).astype(F32)   # (n_mfcc_no_c0+1, T)
		
		M = M[1:, :]  
		
		feats: list[np.ndarray] = [M]

		# Derivadas si corresponde
		if self.config.delta_order >= 1:
			width = 9  
			d1 = librosa.feature.delta(M, width=width, order=1, axis=1, mode="nearest")
			feats.append(d1)

		if self.config.delta_order >= 2:
			d2 = librosa.feature.delta(d1, width=width, order=1, axis=1, mode="nearest")
			feats.append(d2)

		if self.config.delta_order >= 3:
			raise ValueError("Solo se aceptan hasta 2da derivada (delta).")

		return np.concatenate(feats, axis=0).astype(np.float32)  # (C, T)


	# -------------------------------------------------------------------------------------------------  #
	
	@staticmethod
	def _inicio_frames(
		N: int, 
		win: float, 
		hop: float
	) -> np.ndarray:
		last = N - win
		if last < 0:
				return np.array([], dtype=np.int64)
		return np.arange(0, last + 1, hop, dtype=np.int64)

	@staticmethod
	def _calcular_RMS(
		y: np.ndarray, 
		win: float, 
		hop: float
	) -> np.ndarray:
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

	@staticmethod
	def _calcular_ZCR(
		y: np.ndarray, 
		win: float, 
		hop: float
	) -> np.ndarray:
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
