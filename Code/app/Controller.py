from dataclasses import dataclass
from pathlib import Path
import cv2

from Code.image import ImgOrchestrator, OrchestratorCfg
from Code.Estadisticas.BayesAgent import BayesAgent
from Code.audio import AudioPreproc, AudioFeat, Standardizer, KnnModel
from Code.ui.UI import UI
from Code.adapters.Repositorio import Repositorio

@dataclass
class Controller:
    vision: ImgOrchestrator
    bayes: BayesAgent
    knn: KnnModel
    ui: UI
    repo: Repositorio

    def train_kmeans_instance(self) -> None:
        paths = self.repo.list_images(["tornillo", "clavo", "arandela", "tuerca"])
        self.vision.train_from_path(paths=paths)
        self.repo.save_model("vision", "EntrenamientoKMeans", self.vision)

    def run_episode(self) -> None:
        # 1) Tomar/leer 10 imágenes (placeholder: pedir a vision que lo haga)
        X_img, labels_img = self.vision.predict_batch(10)  # define este helper en ImgOrchestrator
        conteo = {c:int(sum(l == c for l in labels_img)) for c in self.vision.class_names}

        # 2) Bayes con hipótesis a-d
        H = [
            [0.25,0.25,0.25,0.25],
            [0.15,0.30,0.30,0.25],
            [0.25,0.35,0.25,0.15],
            [0.50,0.50,0.00,0.00],
        ]
        post = self.bayes.posterior(vecPi=[0.25]*4, Hipotesis_M=H, vecN=list(conteo.values()))
        dec  = self.bayes.decide(post, labels=["a","b","c","d"])

        # 3) Esperar comando de voz y mostrar
        cmd = self._listen_once()
        if cmd == "proporción":
            self.ui.show_proportion(["a","b","c","d"], post.tolist())
        elif cmd == "contar":
            self.ui.show_count(conteo)
        else:
            self.ui.notify("Saliendo.")

    def _listen_once(self) -> str:
        # Ingresa audio, hace preproc+features+std y llama KNN
        # Aquí integrás tu AudioOrchestrator real; este es un stub.
        return "proporción"
