from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Literal, Optional

import csv
import hashlib
import json
import random
import time

import numpy as np

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
AUDIO_EXTENSION = ".wav"


@dataclass(slots=True)
class Repo:
    """
    Filesystem orchestrator for datasets, artifacts and experiment outputs.
    """

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root).expanduser().resolve()

    # ---------- layout ----------
    def ensure_layout(self) -> None:
        """
        Create the default directory structure if missing.
        """
        for rel in (
            "DataBase/data/images1",
            "DataBase/data/images2",
            "DataBase/data/audio",
            "DataBase/data/splits",
            "DataBase/data/indexes",
            "DataBase/models/vision",
            "DataBase/models/audio",
            "DataBase/runs",
            "DataBase/cfg",
            "DataBase/tmp",
        ):
            (self.root / rel).mkdir(parents=True, exist_ok=True)

    # ---------- listing ----------
    def list_images(
            self, 
            number: Literal["1", "2"] = "1",
            labels: Optional[Iterable[str]] = None
            ) -> list[Path]:
        """
        Return sorted image paths under `data/images`, optionally filtered by label folder.
        """
        if number not in {"1", "2"}:
            raise ValueError("No existe ese dataset.")
        
        base = self.root / "DataBase" / "data" / f"images{number}"
        if not base.exists():
            return []

        def valid_image(path: Path) -> bool:
            return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS

        if labels is None:
            return sorted(filter(valid_image, base.rglob("*")))

        labset = {str(label).strip().lower() for label in labels}
        labset.discard("")
        return sorted(
            path
            for path in base.rglob("*")
            if valid_image(path) and path.parent.name.lower() in labset
        )

    def list_audio(self, commands: Optional[Iterable[str]] = None) -> list[Path]:
        """
        Return sorted `.wav` files under `data/audio`, optionally filtered by command folder.
        """
        base = self.root / "data" / "audio"
        if not base.exists():
            return []

        if commands is None:
            return sorted(p for p in base.rglob("*") if p.is_file() and p.suffix.lower() == AUDIO_EXTENSION)

        cmdset = {str(cmd).strip().lower() for cmd in commands}
        cmdset.discard("")
        return sorted(
            path
            for path in base.rglob("*")
            if path.is_file() and path.suffix.lower() == AUDIO_EXTENSION and path.parent.name.lower() in cmdset
        )

    # ---------- splits ----------
    def make_splits(
        self,
        modality: Literal["vision", "audio"],
        train: float = 0.7,
        val: float = 0.15,
        seed: int = 123,
        dataset: Literal["1", "2", "audio"] = "1",
    ) -> Dict[str, int]:
        """
        Create basic train/val/test splits and persist them under `data/splits`.
        """
        if modality not in {"vision", "audio"}:
            raise ValueError("modality debe ser 'vision' o 'audio'")
        if not (0.0 < train < 1.0) or not (0.0 <= val < 1.0) or train + val >= 1.0:
            raise ValueError("Los ratios de train/val deben ser mayores a 0 y su suma menor a 1.")

        if modality == "vision":
            files = self.list_images(number=dataset if dataset in {"1", "2"} else "1")
            if not files:
                raise RuntimeError(f"No se encontraron imágenes en data/images{dataset}.")
        else:
            files = self.list_audio()
            if not files:
                raise RuntimeError("No se encontraron archivos de audio en data/audio.")

        rng = random.Random(seed)
        rng.shuffle(files)
        n = len(files)
        ntr = int(n * train)
        nva = int(n * val)

        parts = {
            "train": files[:ntr],
            "val": files[ntr : ntr + nva],
            "test": files[ntr + nva :],
        }
        for split, paths in parts.items():
            out = self.root / "data" / "splits" / f"{modality}_{split}.txt"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                "\n".join(str(path.relative_to(self.root)) for path in paths),
                encoding="utf-8",
            )
        return {k: len(v) for k, v in parts.items()}

    def iter_split(
        self,
        modality: Literal["vision", "audio"],
        split: Literal["train", "val", "test"],
    ) -> Iterator[Path]:
        """
        Yield absolute paths listed in `data/splits/{modality}_{split}.txt`.
        """
        if modality not in {"vision", "audio"}:
            raise ValueError("modality debe ser 'vision' o 'audio'")
        if split not in {"train", "val", "test"}:
            raise ValueError("split debe ser 'train', 'val' o 'test'")

        path = self.root / "data" / "splits" / f"{modality}_{split}.txt"
        if not path.is_file():
            raise FileNotFoundError(f"No existe el split: {path}")

        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            yield (self.root / line).resolve()

    # ---------- indexes ----------
    def build_index(self, modality: Literal["vision", "audio"]) -> Path:
        """
        Generate a CSV index with relative paths, labels and SHA1 hashes.
        """
        if modality not in {"vision", "audio"}:
            raise ValueError("modality debe ser 'vision' o 'audio'")

        files = self.list_images() if modality == "vision" else self.list_audio()
        idx = self.root / "data" / "indexes" / f"{modality}_index.csv"
        idx.parent.mkdir(parents=True, exist_ok=True)
        with idx.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["relpath", "label", "modality", "sha1"])
            for path in files:
                writer.writerow(
                    [
                        str(path.relative_to(self.root)),
                        path.parent.name.lower(),
                        modality,
                        self._sha1(path),
                    ]
                )
        return idx

    # ---------- models & artifacts ----------
    def save_kmeans(self, name: str, C: np.ndarray) -> Path:
        """
        Persist K-Means centroids under `models/vision/{name}.npz` with key `C`.
        """
        centroides = np.asarray(C, dtype=np.float32)
        if centroides.ndim != 2:
            raise ValueError("C debe ser un arreglo (K, D)")
        out = self._models_dir("vision") / f"{name}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, C=centroides)
        return out

    def load_kmeans(self, name: str) -> Dict[str, np.ndarray]:
        """
        Load K-Means centroids stored with `save_kmeans`.
        """
        path = self._models_dir("vision") / f"{name}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"No se encontró el archivo de centroides: {path}")
        with np.load(path) as data:
            return {k: data[k].astype(np.float32, copy=False) for k in data.files}

    def save_knn(self, name: str, X: np.ndarray, y: np.ndarray) -> Path:
        """Persist audio reference embeddings under `models/audio/{name}.npz`."""
        mat = np.asarray(X, dtype=np.float32)
        if mat.ndim != 2:
            raise ValueError("X debe ser 2D (N, D)")
        labels = np.asarray(y)
        if labels.ndim != 1:
            raise ValueError("y debe ser 1D (N,)")
        if labels.shape[0] != mat.shape[0]:
            raise ValueError("X e y deben tener la misma cantidad de muestras")
        labels = labels.astype(np.str_, copy=False)
        out = self._models_dir("audio") / f"{name}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, X=mat, y=labels)
        return out

    def load_knn(self, name: str) -> Dict[str, np.ndarray]:
        """Load audio reference embeddings stored with `save_knn`."""
        path = self._models_dir("audio") / f"{name}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"No se encontró la base KNN: {path}")
        with np.load(path, allow_pickle=False) as data:
            return {
                "X": data["X"].astype(np.float32, copy=False),
                "y": data["y"].astype(np.str_, copy=False),
            }

    def save_model(self, modality: str, name: str, **arrays: np.ndarray) -> Path:
        """Generic saver kept for backwards compatibility."""
        if not arrays:
            raise ValueError("Debe proveer al menos un arreglo para guardar.")
        out = self._models_dir(modality) / f"{name}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        arrays_np = {k: np.asarray(v) for k, v in arrays.items()}
        np.savez_compressed(out, **arrays_np)
        return out

    def load_model(self, modality: str, name: str) -> Dict[str, np.ndarray]:
        """Generic loader kept for backwards compatibility."""
        path = self._models_dir(modality) / f"{name}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"No existe el modelo: {path}")
        with np.load(path, allow_pickle=False) as data:
            return {k: data[k] for k in data.files}

    def save_json(self, modality: str, name: str, obj: Any) -> Path:
        """Persist JSON metadata under `models/{modality}/{name}.json`."""
        out = self._models_dir(modality) / f"{name}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        return out

    # ---------- runs ----------
    def save_run(
        self,
        run_name: Optional[str],
        metrics: Dict[str, Any],
        posterior: Optional[np.ndarray] = None,
        predictions: Optional[list[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Persist a run folder with metrics, posterior and predictions."""
        if not run_name:
            run_name = time.strftime("%Y-%m-%d_%H-%M-%S")
        base = self.root / "runs" / run_name
        base.mkdir(parents=True, exist_ok=True)
        (base / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if posterior is not None:
            np.savetxt(
                base / "posterior.csv",
                np.asarray(posterior, dtype=np.float64).reshape(1, -1),
                delimiter=",",
                fmt="%.6f",
            )
        if predictions is not None:
            (base / "predictions.csv").write_text(
                "pred\n" + "\n".join(predictions),
                encoding="utf-8",
            )
        if extra:
            (base / "extra.json").write_text(
                json.dumps(extra, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return base

    # ---------- helpers ----------
    def _models_dir(self, modality: str) -> Path:
        allowed = {"vision", "audio"}
        if modality not in allowed:
            raise ValueError(f"modality debe estar en {allowed}")
        return self.root / "models" / modality

    @staticmethod
    def _sha1(path: Path, block: int = 1 << 16) -> str:
        h = hashlib.sha1()
        with path.open("rb") as fh:
            while True:
                chunk = fh.read(block)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()


# Backwards compatibility alias
Repositorio = Repo

__all__ = ["Repo", "Repositorio"]
