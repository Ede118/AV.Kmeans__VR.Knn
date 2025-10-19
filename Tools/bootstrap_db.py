#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import random
import struct
import wave
import zlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

try:
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover
    np = None  # type: ignore

try:
    from PIL import Image, ImageDraw  # type: ignore
except ImportError:  # pragma: no cover
    Image = ImageDraw = None  # type: ignore


CLASS_NAMES = ["arandela", "clavo", "tornillo", "tuerca"]
COMMAND_NAMES = ["proporcion", "contar", "salir"]

IMAGE_COLORS = {
    "arandela": (60, 120, 200),
    "clavo": (190, 60, 80),
    "tornillo": (80, 170, 120),
    "tuerca": (220, 200, 90),
}

COMMAND_FREQS = {
    "proporcion": 660.0,
    "contar": 880.0,
    "salir": 440.0,
}

FORCE = False
GLOBAL_SEED = 42


def ensure_dirs(base: Path) -> Dict[str, Path]:
    base = base.resolve()
    data_dir = base / "data"
    images_dir = data_dir / "images"
    audio_dir = data_dir / "audio"
    models_dir = base / "models"
    models_v = models_dir / "vision"
    models_a = models_dir / "audio"
    runs_dir = base / "runs"

    for path in (base, data_dir, images_dir, audio_dir, models_dir, models_v, models_a, runs_dir):
        path.mkdir(parents=True, exist_ok=True)

    for name in CLASS_NAMES:
        (images_dir / name).mkdir(parents=True, exist_ok=True)

    for name in COMMAND_NAMES:
        (audio_dir / name).mkdir(parents=True, exist_ok=True)

    return {
        "base": base,
        "data": data_dir,
        "images": images_dir,
        "audio": audio_dir,
        "models_v": models_v,
        "models_a": models_a,
        "runs": runs_dir,
    }


def _should_skip(path: Path) -> bool:
    if FORCE:
        return False
    return path.exists() and path.stat().st_size > 0


def _generate_png_bytes(width: int, height: int, color: Tuple[int, int, int]) -> bytes:
    raw = bytearray()
    for _ in range(height):
        raw.append(0)
        raw.extend(color)

    def chunk(chunk_type: bytes, data: bytes) -> bytes:
        checksum = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
        return (
            struct.pack("!I", len(data))
            + chunk_type
            + data
            + struct.pack("!I", checksum)
        )

    header = struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)
    compressed = zlib.compress(bytes(raw))

    png = bytearray(b"\x89PNG\r\n\x1a\n")
    png.extend(chunk(b"IHDR", header))
    png.extend(chunk(b"IDAT", compressed))
    png.extend(chunk(b"IEND", b""))
    return bytes(png)


def _write_png_with_pillow(path: Path, color: Tuple[int, int, int], rng: random.Random) -> None:
    assert Image is not None and ImageDraw is not None
    width = height = 160
    img = Image.new("RGB", (width, height), color=color)
    draw = ImageDraw.Draw(img)

    inset = rng.randint(15, 35)
    rect = (inset, inset, width - inset, height - inset)
    outline_color = tuple(max(0, min(255, c + rng.randint(-30, 30))) for c in color)
    draw.rectangle(rect, outline=outline_color, width=3)

    line_y = rng.randint(20, height - 20)
    line_color = tuple(max(0, min(255, c + rng.randint(-40, 20))) for c in color)
    draw.line([(10, line_y), (width - 10, height - line_y)], fill=line_color, width=4)
    img.save(path, format="PNG")


def _write_png_fallback(path: Path, color: Tuple[int, int, int]) -> None:
    pixel = _generate_png_bytes(1, 1, color)
    encoded = base64.b64encode(pixel)
    path.write_bytes(base64.b64decode(encoded))


def populate_images(images_dir: Path, per_class: int, seed: int) -> int:
    rng = random.Random(seed)
    for class_name in CLASS_NAMES:
        class_dir = images_dir / class_name
        class_title = class_name.capitalize()
        color = IMAGE_COLORS[class_name]
        for idx in range(1, per_class + 1):
            filename = f"{class_title}{idx:03d}.png"
            filepath = class_dir / filename
            if _should_skip(filepath):
                continue
            if Image is not None and ImageDraw is not None:
                _write_png_with_pillow(filepath, color, rng)
            else:
                _write_png_fallback(filepath, color)
    return sum(1 for _ in images_dir.glob("*/*.png"))


def _sine_wave(sample_rate: int, duration: float, frequency: float) -> Iterable[int]:
    total_samples = int(sample_rate * duration)
    for i in range(total_samples):
        sample = math.sin(2 * math.pi * frequency * (i / sample_rate))
        yield int(sample * 32767 * 0.6)


def populate_audio(audio_dir: Path, per_cmd: int, seed: int) -> int:
    rng = random.Random(seed + 17)
    sample_rate = 16_000
    for cmd in COMMAND_NAMES:
        cmd_dir = audio_dir / cmd
        freq_base = COMMAND_FREQS[cmd]
        title = cmd.capitalize()
        for idx in range(1, per_cmd + 1):
            filename = f"{title}{idx:03d}.wav"
            filepath = cmd_dir / filename
            if _should_skip(filepath):
                continue

            variation = rng.uniform(-0.03, 0.03)
            duration = rng.uniform(0.5, 0.6)
            frequency = freq_base * (1.0 + variation)

            with wave.open(str(filepath), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                frames = bytearray()
                for sample in _sine_wave(sample_rate, duration, frequency):
                    frames.extend(struct.pack("<h", sample))
                wav_file.writeframes(frames)

    return sum(1 for _ in audio_dir.glob("*/*.wav"))


def _write_lines(path: Path, lines: Sequence[str]) -> int:
    if _should_skip(path):
        existing = path.read_text(encoding="utf-8").splitlines()
        return len(existing)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(f"{line}\n")
    return len(lines)


def _split_items(items: List[str], rng: random.Random) -> Tuple[List[str], List[str], List[str]]:
    rng.shuffle(items)
    total = len(items)
    train_end = max(1, int(total * 0.7)) if total >= 3 else max(1, total - 2)
    val_end = train_end + max(1, int(total * 0.15)) if total >= 3 else train_end + 1
    if val_end > total:
        val_end = total
    train = items[:train_end]
    val = items[train_end:val_end]
    test = items[val_end:]
    if not test and total >= 3:
        test = [train.pop()]
    return train, val, test


def make_splits_txt(data_dir: Path, seed: int) -> Dict[str, Tuple[int, int, int]]:
    images_dir = data_dir / "images"
    audio_dir = data_dir / "audio"

    image_paths = sorted(
        str(path.relative_to(data_dir))
        for path in images_dir.glob("*/*.png")
    )
    audio_paths = sorted(
        str(path.relative_to(data_dir))
        for path in audio_dir.glob("*/*.wav")
    )

    vision_rng = random.Random(seed + 101)
    audio_rng = random.Random(seed + 202)

    v_train, v_val, v_test = _split_items(image_paths.copy(), vision_rng)
    a_train, a_val, a_test = _split_items(audio_paths.copy(), audio_rng)

    counts = {}
    counts["vision"] = (
        _write_lines(data_dir / "vision_train.txt", v_train),
        _write_lines(data_dir / "vision_val.txt", v_val),
        _write_lines(data_dir / "vision_test.txt", v_test),
    )
    counts["audio"] = (
        _write_lines(data_dir / "audio_train.txt", a_train),
        _write_lines(data_dir / "audio_val.txt", a_val),
        _write_lines(data_dir / "audio_test.txt", a_test),
    )
    return counts


def _write_json(path: Path, payload: object) -> None:
    if _should_skip(path):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_models(models_v: Path, models_a: Path) -> None:
    if np is not None:
        rng = np.random.default_rng(GLOBAL_SEED)

        centroids = rng.normal(loc=0.0, scale=1.0, size=(4, 8)).astype("float32")
        centroids_path = models_v / "kmeans_centroids.npz"
        if not _should_skip(centroids_path):
            np.savez(centroids_path, C=centroids)

        mapping_path = models_v / "cluster_to_class.json"
        _write_json(mapping_path, {str(i): name for i, name in enumerate(CLASS_NAMES)})

        mean = np.linspace(-1.0, 1.0, 20, dtype="float32")
        std = np.linspace(0.5, 1.5, 20, dtype="float32")
        standardizer_path = models_a / "standardizer.npz"
        if not _should_skip(standardizer_path):
            np.savez(standardizer_path, mean=mean, std=std)

        X = rng.normal(size=(3, 20)).astype("float32")
        y = np.array(COMMAND_NAMES, dtype="<U16")
        knn_path = models_a / "knn_model.npz"
        if not _should_skip(knn_path):
            np.savez(knn_path, X=X, y=y)
    else:
        mapping_path = models_v / "cluster_to_class.json"
        _write_json(mapping_path, {str(i): name for i, name in enumerate(CLASS_NAMES)})

        standardizer_json = models_a / "standardizer.json"
        if not _should_skip(standardizer_json):
            payload = {
                "mean": [round(-1.0 + i * 0.105, 6) for i in range(20)],
                "std": [round(0.5 + i * 0.05, 6) for i in range(20)],
            }
            _write_json(standardizer_json, payload)

        knn_json = models_a / "knn_model.json"
        if not _should_skip(knn_json):
            payload = {
                "X": [
                    [round(math.sin((i + j + 1) * 0.3), 6) for j in range(20)]
                    for i in range(3)
                ],
                "y": COMMAND_NAMES,
            }
            _write_json(knn_json, payload)


def save_sample_run(runs_dir: Path) -> Path:
    base_dt = datetime(2024, 1, 1, 12, 0, 0)
    run_dt = base_dt + timedelta(seconds=GLOBAL_SEED)
    run_name = run_dt.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = run_dir / "metrics.json"
    metrics_payload = {
        "silhouette": 0.42,
        "acc_val": 0.80,
        "n_piezas": 10,
        "seed": GLOBAL_SEED,
    }
    _write_json(metrics_path, metrics_payload)

    posterior_path = run_dir / "posterior.csv"
    if not _should_skip(posterior_path):
        with posterior_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([0.10, 0.15, 0.20, 0.55])

    predictions_path = run_dir / "predictions_images.csv"
    if not _should_skip(predictions_path):
        rng = random.Random(GLOBAL_SEED + 303)
        labels = [rng.choice(CLASS_NAMES) for _ in range(10)]
        with predictions_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["pred"])
            for label in labels:
                writer.writerow([label])

    log_path = run_dir / "log.txt"
    if not _should_skip(log_path):
        lines = [
            "[info] bootstrap run initialized",
            "[info] loading centroids and embeddings",
            "[info] evaluation completed successfully",
            "[info] artifacts stored on disk",
        ]
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap DataBase assets.")
    parser.add_argument("--base", type=Path, default=Path("DataBase"), help="Ruta base del dataset.")
    parser.add_argument("--imgs", type=int, default=12, help="Cantidad de imágenes por clase.")
    parser.add_argument("--wavs", type=int, default=8, help="Cantidad de audios por comando.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria.")
    parser.add_argument("--force", action="store_true", help="Sobrescribir archivos existentes.")
    args = parser.parse_args()

    global FORCE, GLOBAL_SEED
    FORCE = args.force
    GLOBAL_SEED = args.seed

    paths = ensure_dirs(args.base)

    images_total = populate_images(paths["images"], args.imgs, args.seed)
    audio_total = populate_audio(paths["audio"], args.wavs, args.seed)
    splits = make_splits_txt(paths["data"], args.seed)
    save_models(paths["models_v"], paths["models_a"])
    run_dir = save_sample_run(paths["runs"])

    vision_counts = splits["vision"]
    audio_counts = splits["audio"]

    base_abs = paths["base"].resolve()
    print(f"[ok] Estructura creada en: {base_abs}")
    print(f"[ok] Imágenes: {images_total}")
    print(f"[ok] Audios:   {audio_total}")
    print(
        "[ok] Splits:   "
        f"vision(train/val/test)={vision_counts[0]}/{vision_counts[1]}/{vision_counts[2]}  "
        f"audio(train/val/test)={audio_counts[0]}/{audio_counts[1]}/{audio_counts[2]}"
    )
    if np is not None:
        print("[ok] Modelos:  vision=C(4x8) audio=X(3x20)")
    else:
        print("[ok] Modelos:  vision=json audio=json")
    print(f"[ok] Run demo: {run_dir.resolve()}")


if __name__ == "__main__":
    main()

# Comandos de prueba:
# python tools/bootstrap_db.py --base DataBase --imgs 12 --wavs 8 --seed 42
# python tools/bootstrap_db.py --base DataBase --force
