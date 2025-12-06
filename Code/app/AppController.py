from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys

import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Code.image import ImgOrchestrator
from Code.audio import AudioOrchestrator
from Code.Estadisticas import BayesAgent


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Database" / "data"
MODELS_DIR = PROJECT_ROOT / "Database" / "models"


@dataclass(frozen=False)
class AppController:
	IOrch: ImgOrchestrator = field(default_factory=ImgOrchestrator)
	AOrch: AudioOrchestrator = field(default_factory=AudioOrchestrator)
	Bayes: BayesAgent = field(default_factory=BayesAgent)
	_last_img_df: pd.DataFrame | None = None

	# Parámetros por defecto para Bayes (cajas a, b, c, d)
	bayes_pi: np.ndarray = field(init=False)
	bayes_P: np.ndarray = field(init=False)
	bayes_labels: list[str] = field(init=False)

	def __post_init__(self) -> None:
		# --- Parámetros por defecto de Bayes ---
		# Matriz P[k,i] = P(categoría i | hipótesis k)
		# Este es exactamente el ejemplo de las cajas.
		self.bayes_P = np.array([
			[250, 250, 250, 250],   # caja a)
			[150, 300, 300, 250],   # caja b)
			[250, 350, 250, 150],   # caja c)
			[500, 500,   0,   0],   # caja d)
		], dtype=float) / 1000.0   # cada fila suma 1

		# Prior uniforme sobre las 4 hipótesis
		K = self.bayes_P.shape[0]
		self.bayes_pi = np.ones(K, dtype=float) / K

		# Labels de las hipótesis (puedes cambiarlos por algo más lindo luego)
		self.bayes_labels = ["a", "b", "c", "d"]

		try:
			self.IOrch.cargar_modelo("Database/models/kmeans.npz")
		except Exception as exc:
			print(f"[AppController] No se pudo cargar modelo de imagen: {exc!r}")
		try:
			self.AOrch.cargar_modelo()  # audio sí tiene ruta por defecto
		except Exception as exc:
			print(f"[AppController] No se pudo cargar modelo de audio: {exc!r}")
		


	
	def predecir_img(self, ruta_imagen: Path) -> dict:
		"""
		Analiza una imagen utilizando el orquestador de imágenes.
		Devuelve un solo diccionario con los resultados del análisis.
		"""
		resultados, _ = self.IOrch.predecir(
			ruta_imagen,
			recursive=False,
			return_df=True,
		)
		if not resultados:
			raise RuntimeError(f"No se pudo obtener predicción para la imagen: {ruta_imagen}")
		return resultados[0]

	def predecir_carpeta_img(self, ruta_carpeta: Path) -> pd.DataFrame:
		"""
		Analiza todas las imágenes en una carpeta utilizando el orquestador de imágenes.
		Devuelve solo el DataFrame con los resultados del análisis.
		"""
		_, df = self.IOrch.predecir(
			ruta_carpeta,
			recursive=True,
			return_df=True,
		)
		return df
	
	def clasificar_carpeta_img_df(self, ruta_carpeta: Path) -> pd.DataFrame:
		"""
		Clasifica todas las imágenes de `ruta_carpeta` y devuelve un DataFrame
		con columnas estandarizadas:

		    id, path, cluster, label

		Además guarda el DF en self._last_img_df para reutilizarlo en otros plots.
		"""
		ruta_carpeta = Path(ruta_carpeta)

		# Usamos directamente el orquestador y pedimos también el DataFrame.
		_, df_raw = self.IOrch.predecir(
			ruta_carpeta,
			recursive=True,
			return_df=True,
		)

		# Renombrar columnas desde la salida original:
		# "id", "path relativo", "Número de Cluster", "Clasificación"
		df = df_raw.rename(
			columns={
				"id": "ID",
				"path relativo": "path",
			}
		)

		# Nos quedamos exactamente con estas cuatro columnas (en este orden)
		df = df[["ID", "path", "Número de Cluster", "Clasificación"]].copy()

		self._last_img_df = df
		return df

	def make_kmeans_features_3d_figure(self, carpeta: Path) -> plt.Figure:
		"""
		Genera un gráfico 3D de las features de KMeans para todas las imágenes
		de `carpeta`.

		- Ejes: (r_hull, radial_var, huecos), cada uno en [0, 1] aprox.
		- Puntos: imágenes clasificadas, coloreadas por cluster.
		- Marcadores 'X': centroides de KMeans en ese mismo espacio.
		"""
		carpeta = Path(carpeta)

		data = self.IOrch.extraer_features_3d_para_directorio(
			carpeta,
			recursive=True,
		)

		X: np.ndarray = data["X"]               # (N, 3)
		clusters: np.ndarray = data["clusters"] # (N,)
		centroids: np.ndarray = data["centroids"]  # (K, 3)

		fig = plt.figure(figsize=(6, 5))
		ax = fig.add_subplot(111, projection="3d")

		# Puntos (imágenes): tamaño moderado, colores por cluster
		sc = ax.scatter(
			X[:, 0],
			X[:, 1],
			X[:, 2],
			c=clusters,
			s=30,
			alpha=0.95,
			cmap="viridis",
		)

		# Centroides: más finos y bien contrastados
		ax.scatter(
			centroids[:, 0],
			centroids[:, 1],
			centroids[:, 2],
			marker="X",
			s=60,              # antes 120 o 140, ahora más chico/“fino”
			c="red",           # color fijo para destacar
			edgecolor="black",
			linewidth=1.0,
		)

		ax.set_title("KMeans en espacio de features (r_hull, radial_var, huecos)")
		ax.set_xlabel("r_hull")
		ax.set_ylabel("radial_var")
		ax.set_zlabel("huecos")

		# Como tus features están normalizadas en [0, 1], fijamos límites
		ax.set_xlim(0.0, 1.0)
		ax.set_ylim(0.0, 1.0)
		ax.set_zlim(0.0, 1.0)

		# Colorbar opcional, por cluster
		cbar = fig.colorbar(sc, ax=ax, shrink=0.7)
		cbar.set_label("Cluster")

		return fig


	def get_last_img_df(self) -> pd.DataFrame | None:
		"""
		Devuelve el último DataFrame de imágenes clasificado
		(o None si todavía no se clasificó ninguna carpeta).
		"""
		return self._last_img_df

	def _contar_por_cluster(
		self,
		df: pd.DataFrame,
		column: str | None = None,
		num_categories: int | None = None,
	) -> np.ndarray:
		"""
		Devuelve un vector n de longitud C con los conteos por cluster.

		- column: nombre de la columna con el cluster. Si es None,
			intenta adivinarlo entre varios nombres típicos.
		"""
		if column is None:
			candidatos = ["cluster", "Cluster", "Número de Cluster", "Número de Cluste"]
			for c in candidatos:
				if c in df.columns:
					column = c
					break
		if column is None or column not in df.columns:
			raise ValueError(
				f"No se encontró columna de cluster en el DataFrame. "
				f"Columnas disponibles: {list(df.columns)}"
			)

		clusters = df[column].to_numpy(dtype=int)

		if num_categories is None:
			num_categories = int(clusters.max()) + 1

		n = np.bincount(clusters, minlength=num_categories)
		return n

	def new_bayes(
		self,
		df: pd.DataFrame,
		pi: np.ndarray,
		P: np.ndarray,
		labels_hipotesis: list[str],
		*,
		use_logs: bool = True,
		strict_zeros: bool = True,
	) -> tuple[np.ndarray, str | list[str]]:
		"""
		Aplica Bayes usando:
		- df["cluster"] como categorías observadas,
		- prior pi (shape (K,)),
		- matriz de likelihood P (shape (K, C)).

		Devuelve:
		- posterior: np.ndarray shape (K,)
		- decision : etiqueta más probable (o lista si configuras policy='all')
		"""
		pi = np.asarray(pi, float).reshape(-1)
		P = np.asarray(P, float)

		K, C = P.shape
		if pi.shape[0] != K:
			raise ValueError(f"pi debe tener tamaño {K}, vino {pi.shape[0]}")
		if len(labels_hipotesis) != K:
			raise ValueError("Cantidad de labels de hipótesis no coincide con K")

		# 1) Conteos por cluster
		n = self._contar_por_cluster(df, column="cluster", num_categories=C)

		# 2) Posterior con BayesAgent
		post = self.Bayes.posterior(pi, P, n, use_logs=use_logs, strict_zeros=strict_zeros)

		# 3) Decisión
		decision = self.Bayes.decide(post, labels=labels_hipotesis)

		OUTPUT_DIR = PROJECT_ROOT / "Database" / "output"
		OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

		np.save(OUTPUT_DIR / "bayes_posterior.npy", posterior)
		df.to_csv(OUTPUT_DIR / "ultima_carpeta_clasificada.csv", index=False)

		return post, decision

	def bayes(
		self,
		pi: np.ndarray,
		P: np.ndarray,
		labels_hipotesis: list[str],
		*,
		use_logs: bool = True,
		strict_zeros: bool = True,
	) -> tuple[np.ndarray, str | list[str]]:
		"""
		Igual que bayes_desde_df_clusters, pero usando self._last_img_df.

		Lanza RuntimeError si todavía no se clasificó ninguna carpeta.
		"""
		if self._last_img_df is None:
			raise RuntimeError("No hay ningún DataFrame de imágenes cargado (self._last_img_df es None).")

		return self.new_bayes(
			self._last_img_df,
			pi,
			P,
			labels_hipotesis,
			use_logs=use_logs,
			strict_zeros=strict_zeros,
		)

	def bayes_desde_df_clusters(
		self,
		df: pd.DataFrame,
		pi: np.ndarray,
		P: np.ndarray,
		labels_hipotesis: list[str],
		*,
		column: str | None = None,
		use_logs: bool = True,
		strict_zeros: bool = True,
	) -> tuple[np.ndarray, str | list[str]]:
		"""
		Aplica Bayes usando los clusters del DataFrame como categorías observadas.

		- df: DataFrame con una columna de cluster.
		- pi: prior sobre hipótesis, shape (K,)
		- P: matriz de likelihood, shape (K, C)
		- labels_hipotesis: nombres de las hipótesis (len K)
		- column: nombre de la columna con el cluster (si None, se infiere).
		"""
		pi = np.asarray(pi, float).reshape(-1)
		P = np.asarray(P, float)

		K, C = P.shape
		if pi.shape[0] != K:
			raise ValueError(f"pi debe tener tamaño {K}, vino {pi.shape[0]}")
		if len(labels_hipotesis) != K:
			raise ValueError(
				f"labels_hipotesis debe tener {K} elementos, vino {len(labels_hipotesis)}"
			)

		# 1) conteos por cluster
		n = self._contar_por_cluster(df, column=column, num_categories=C)

		# 2) posterior con BayesAgent
		post = self.Bayes.posterior(pi, P, n, use_logs=use_logs, strict_zeros=strict_zeros)

		# 3) decisión
		decision = self.Bayes.decide(post, labels=labels_hipotesis)

		return post, decision


	def grabar_audio(self, *, duracion_segundos: float, ruta_salida: Path) -> None:
		"""
		Comienza una grabación de audio por la duración especificada y guarda el archivo en la ruta dada.
		"""
		self.AOrch.grabar_audio(dur_sec=duracion_segundos, salida=ruta_salida)
		return None
	
	def predecir_audio(self, ruta_audio: Path) -> dict:
		"""
		Analiza un archivo de audio utilizando el orquestador de audio.
		Devuelve un diccionario con los resultados del análisis.
		"""
		resultados = self.AOrch.predecir_comando(ruta_audio)
		return resultados

