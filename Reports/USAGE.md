# Ejemplos de uso

## Visión
```python
from pathlib import Path
from Code.vision import ImgOrchestrator

paths = [
    Path("DataBase/images/tornillo/img01.png"),
    Path("DataBase/images/tuerca/img05.png"),
]

orch = ImgOrchestrator()
orch.fit_from_paths(paths, k=4, seed=42)

pred = orch.identify_path("DataBase/images/tornillo/img09.png")
print(pred)
```

## Persistencia de centroides con Repo
```python
import numpy as np
from pathlib import Path
from Code.adapters import Repo

repo = Repo(Path("DataBase"))
centroids = np.load("runs/vision_last_centroids.npz")["C"]
repo.save_kmeans("baseline", centroids.astype(np.float32))

loaded = repo.load_kmeans("baseline")["C"]
```

## Audio (KNN)
```python
from pathlib import Path
from Code.adapters import Repo
from Code.audio import AudioOrchestrator

paths = [
    Path("DataBase/audio/forward/001.wav"),
    Path("DataBase/audio/left/002.wav"),
]
labels = ["forward", "left"]

orch = AudioOrchestrator()
orch.build_reference_from_paths(paths, labels)

repo = Repo(Path("DataBase"))
orch.save_reference_to_repo(repo, "commands")
orch.load_reference_from_repo(repo, "commands")

pred = orch.identify_path("DataBase/audio/forward/010.wav")
print(pred)
```
