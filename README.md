#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_dataset.py
==================

Назначение
----------
Один скрипт для обслуживания YOLO-датасета:

    cleanup   удалить лишние labels, для которых нет изображения;
    annotate  создать отсутствующие labels с помощью container.pt и text.pt;
    split     создать обучающие датасеты container и text;
    all       выполнить cleanup -> annotate -> split.

Исходная структура:

master_dataset/
├── images/
│   ├── vertical/
│   ├── oneline/
│   ├── twolines/
│   ├── unknown/
│   └── yantar/
│       ├── vertical/
│       ├── oneline/
│       ├── twolines/
│       └── unknown/
└── labels/
    ├── vertical/
    │   ├── container.pt
    │   ├── text.pt
    │   ├── classes.txt
    │   └── *.txt
    ├── oneline/
    ├── twolines/
    ├── unknown/
    └── yantar/
        ├── vertical/
        ├── oneline/
        ├── twolines/
        └── unknown/

Исходные классы:
    0 container_bbox
    1 text

Веса container.pt и text.pt ищутся в соответствующей папке labels.
Например для images/yantar/vertical/a.jpg используются:
    labels/yantar/vertical/container.pt
    labels/yantar/vertical/text.pt

Установка зависимостей:
    pip install ultralytics opencv-python

Запуск:
    python prepare_dataset.py cleanup --dataset-root master_dataset
    python prepare_dataset.py annotate --dataset-root master_dataset
    python prepare_dataset.py annotate --dataset-root master_dataset --overwrite
    python prepare_dataset.py split --dataset-root master_dataset --output split_datasets
    python prepare_dataset.py all --dataset-root master_dataset --output split_datasets

Результат split:

split_datasets/
├── container/
│   ├── images/train/
│   ├── images/val/
│   ├── labels/train/
│   ├── labels/val/
│   ├── classes.txt
│   └── data.yaml
└── text/
    ├── vertical/
    │   ├── images/train/
    │   ├── images/val/
    │   ├── labels/train/
    │   ├── labels/val/
    │   ├── classes.txt
    │   └── data.yaml
    ├── oneline/
    └── twolines/

Правила split:
    - unknown и yantar/unknown не используются;
    - container объединяет vertical, oneline, twolines и yantar-варианты;
    - container labels содержат только класс 0 container_bbox;
    - изображения без container_bbox включаются в container с пустым label;
    - text создаётся отдельно для vertical/oneline/twolines;
    - text-изображение — crop container_bbox, увеличенный на 25% по ширине
      и высоте (по 12.5% с каждой стороны при --expand 0.25);
    - исходные классы 1 пересчитываются относительно crop и записываются
      как класс 0 text;
    - сначала собираются валидные text crops, затем regular и yantar
      независимо делятся на train/val, чтобы сохранять их пропорцию;
    - classes.txt никогда не удаляется.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
ALL_TYPES = ("vertical", "oneline", "twolines", "unknown")
TRAIN_TYPES = ("vertical", "oneline", "twolines")
SOURCES = ("regular", "yantar")
PROTECTED_TEXT_FILES = {"classes.txt"}


@dataclass(frozen=True)
class Group:
    number_type: str
    source: str
    image_dir: Path
    label_dir: Path


@dataclass(frozen=True)
class SourceSample:
    number_type: str
    source: str
    image_path: Path
    label_path: Path


@dataclass(frozen=True)
class TextCrop:
    number_type: str
    source: str
    image_path: Path
    label_lines: tuple[str, ...]
    name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Подготовка YOLO-датасета: cleanup, annotate, split или all.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("stage", choices=("cleanup", "annotate", "split", "all"))
    parser.add_argument("--dataset-root", type=Path, default=Path("master_dataset"))
    parser.add_argument("--output", type=Path, default=Path("split_datasets"))
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--expand", type=float, default=0.25)
    parser.add_argument("--container-conf", type=float, default=0.25)
    parser.add_argument("--text-conf", type=float, default=0.25)
    parser.add_argument("--device", default=None, help="Например cpu, 0 или 0,1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--copy-mode", choices=("copy", "hardlink"), default="copy")
    parser.add_argument("--keep-output", action="store_true")
    parser.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO"
    )
    args = parser.parse_args()

    if not 0.0 < args.train_ratio < 1.0:
        parser.error("--train-ratio должен быть между 0 и 1")
    if args.expand < 0:
        parser.error("--expand не может быть отрицательным")
    if not 0 <= args.container_conf <= 1:
        parser.error("--container-conf должен быть между 0 и 1")
    if not 0 <= args.text_conf <= 1:
        parser.error("--text-conf должен быть между 0 и 1")
    return args


def get_group(root: Path, number_type: str, source: str) -> Group:
    if source == "regular":
        return Group(number_type, source, root / "images" / number_type, root / "labels" / number_type)
    return Group(
        number_type,
        source,
        root / "images" / "yantar" / number_type,
        root / "labels" / "yantar" / number_type,
    )


def all_groups(root: Path, types: Sequence[str] = ALL_TYPES):
    for number_type in types:
        for source in SOURCES:
            yield get_group(root, number_type, source)


def list_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def safe_name(number_type: str, source: str, stem: str) -> str:
    return f"{source}__{number_type}__{stem}"


def write_label(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines)
    path.write_text(content + ("\n" if content else ""), encoding="utf-8")


def cleanup_orphan_labels(root: Path) -> int:
    removed = 0
    for group in all_groups(root):
        if not group.label_dir.is_dir():
            continue
        stems = {p.stem for p in list_images(group.image_dir)}
        for txt in sorted(group.label_dir.glob("*.txt")):
            if txt.name.lower() in PROTECTED_TEXT_FILES:
                continue
            if txt.stem not in stems:
                txt.unlink()
                removed += 1
                logging.info("Удалён label без изображения: %s", txt)
    logging.info("Cleanup завершён. Удалено: %d", removed)
    return removed


class ModelCache:
    def __init__(self) -> None:
        self.models: dict[Path, object] = {}

    def get(self, weights: Path):
        key = weights.resolve()
        if key not in self.models:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError("Установите ultralytics: pip install ultralytics") from exc
            logging.info("Загрузка весов: %s", weights)
            self.models[key] = YOLO(str(weights))
        return self.models[key]


def predict_xyxy(model, image, conf: float, device: Optional[str]):
    kwargs = {"source": image, "conf": conf, "verbose": False}
    if device is not None:
        kwargs["device"] = device
    results = model.predict(**kwargs)
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return []
    boxes = results[0].boxes
    coords = boxes.xyxy.detach().cpu().tolist()
    scores = boxes.conf.detach().cpu().tolist()
    return [(*map(float, box), float(score)) for box, score in zip(coords, scores)]


def clip_box(box: Sequence[float], width: int, height: int):
    x1, y1, x2, y2 = map(float, box)
    return (
        max(0.0, min(float(width), x1)),
        max(0.0, min(float(height), y1)),
        max(0.0, min(float(width), x2)),
        max(0.0, min(float(height), y2)),
    )


def expand_box(box: Sequence[float], width: int, height: int, ratio: float):
    x1, y1, x2, y2 = clip_box(box, width, height)
    pad_x = max(0.0, x2 - x1) * ratio / 2.0
    pad_y = max(0.0, y2 - y1) * ratio / 2.0
    x1, y1, x2, y2 = clip_box(
        (x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), width, height
    )
    return int(x1), int(y1), min(width, int(x2 + 0.999999)), min(height, int(y2 + 0.999999))


def xyxy_to_yolo(box: Sequence[float], width: int, height: int, class_id: int) -> Optional[str]:
    x1, y1, x2, y2 = clip_box(box, width, height)
    if x2 <= x1 or y2 <= y1 or width <= 0 or height <= 0:
        return None
    xc = (x1 + x2) / 2.0 / width
    yc = (y1 + y2) / 2.0 / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def yolo_to_xyxy(line: str, width: int, height: int):
    parts = line.split()
    if len(parts) != 5:
        return None
    try:
        class_id = int(float(parts[0]))
        xc, yc, bw, bh = map(float, parts[1:])
    except ValueError:
        return None
    return (
        class_id,
        *clip_box(
            (
                (xc - bw / 2.0) * width,
                (yc - bh / 2.0) * height,
                (xc + bw / 2.0) * width,
                (yc + bh / 2.0) * height,
            ),
            width,
            height,
        ),
    )


def read_labels(path: Path, width: int, height: int):
    if not path.is_file():
        return []
    result = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parsed = yolo_to_xyxy(raw.strip(), width, height)
        if parsed is not None:
            result.append(parsed)
    return result


def choose_container(labels):
    boxes = [item[1:] for item in labels if item[0] == 0]
    if not boxes:
        return None
    return max(boxes, key=lambda b: max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]))


def annotate_missing_labels(
    root: Path,
    expand_ratio: float,
    container_conf: float,
    text_conf: float,
    device: Optional[str],
    overwrite: bool,
):
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Установите opencv-python: pip install opencv-python") from exc

    cache = ModelCache()
    stats = {"processed": 0, "skipped": 0, "empty": 0, "container_only": 0, "with_text": 0}

    for group in all_groups(root):
        group.label_dir.mkdir(parents=True, exist_ok=True)
        container_weights = group.label_dir / "container.pt"
        text_weights = group.label_dir / "text.pt"

        for image_path in list_images(group.image_dir):
            label_path = group.label_dir / f"{image_path.stem}.txt"
            if label_path.exists() and not overwrite:
                stats["skipped"] += 1
                continue
            stats["processed"] += 1

            if not container_weights.is_file():
                write_label(label_path, [])
                stats["empty"] += 1
                logging.warning("Нет %s; создан пустой %s", container_weights, label_path)
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                write_label(label_path, [])
                stats["empty"] += 1
                logging.error("Не удалось прочитать %s", image_path)
                continue

            h, w = image.shape[:2]
            detections = predict_xyxy(cache.get(container_weights), image, container_conf, device)
            if not detections:
                write_label(label_path, [])
                stats["empty"] += 1
                continue

            container = max(detections, key=lambda x: x[4])[:4]
            lines: list[str] = []
            container_line = xyxy_to_yolo(container, w, h, 0)
            if container_line:
                lines.append(container_line)

            if not text_weights.is_file():
                write_label(label_path, lines)
                stats["container_only"] += 1
                continue

            x1, y1, x2, y2 = expand_box(container, w, h, expand_ratio)
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                write_label(label_path, lines)
                stats["container_only"] += 1
                continue

            text_detections = predict_xyxy(cache.get(text_weights), crop, text_conf, device)
            text_detections.sort(key=lambda x: (x[1], x[0]))
            for tx1, ty1, tx2, ty2, _ in text_detections:
                line = xyxy_to_yolo((tx1 + x1, ty1 + y1, tx2 + x1, ty2 + y1), w, h, 1)
                if line:
                    lines.append(line)

            write_label(label_path, lines)
            if len(lines) > 1:
                stats["with_text"] += 1
            else:
                stats["container_only"] += 1

    logging.info("Annotate завершён: %s", stats)
    return stats


def split_list(items: Sequence, train_ratio: float, rng: random.Random):
    values = list(items)
    rng.shuffle(values)
    if not values:
        return [], []
    if len(values) == 1:
        return values, []
    train_count = round(len(values) * train_ratio)
    train_count = max(1, min(len(values) - 1, train_count))
    return values[:train_count], values[train_count:]


def transfer_file(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    if mode == "hardlink":
        try:
            os.link(source, destination)
            return
        except OSError as exc:
            logging.warning("Hardlink не создан (%s), копирование: %s", exc, source)
    shutil.copy2(source, destination)


def prepare_dataset_dir(root: Path) -> None:
    for subset in ("train", "val"):
        (root / "images" / subset).mkdir(parents=True, exist_ok=True)
        (root / "labels" / subset).mkdir(parents=True, exist_ok=True)


def write_metadata(root: Path, class_name: str) -> None:
    (root / "classes.txt").write_text(class_name + "\n", encoding="utf-8")
    (root / "data.yaml").write_text(
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n\n"
        "names:\n"
        f"  0: {class_name}\n",
        encoding="utf-8",
    )


def collect_source_samples(root: Path, number_type: str, source: str):
    group = get_group(root, number_type, source)
    return [
        SourceSample(number_type, source, image_path, group.label_dir / f"{image_path.stem}.txt")
        for image_path in list_images(group.image_dir)
    ]


def create_container_dataset(
    root: Path,
    output_root: Path,
    train_ratio: float,
    seed: int,
    copy_mode: str,
):
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Установите opencv-python: pip install opencv-python") from exc

    destination = output_root / "container"
    prepare_dataset_dir(destination)
    stats = {"train": 0, "val": 0, "empty_labels": 0, "read_errors": 0}

    # Стратификация одновременно по типу и источнику.
    for type_index, number_type in enumerate(TRAIN_TYPES):
        for source_index, source in enumerate(SOURCES):
            samples = collect_source_samples(root, number_type, source)
            rng = random.Random(seed + type_index * 1000 + source_index)
            train, val = split_list(samples, train_ratio, rng)

            for subset, subset_samples in (("train", train), ("val", val)):
                for sample in subset_samples:
                    image = cv2.imread(str(sample.image_path))
                    if image is None:
                        stats["read_errors"] += 1
                        logging.error("Не удалось прочитать %s", sample.image_path)
                        continue
                    h, w = image.shape[:2]
                    container = choose_container(read_labels(sample.label_path, w, h))
                    lines: list[str] = []
                    if container is not None:
                        line = xyxy_to_yolo(container, w, h, 0)
                        if line:
                            lines.append(line)
                    if not lines:
                        stats["empty_labels"] += 1

                    name = safe_name(sample.number_type, sample.source, sample.image_path.stem)
                    dst_image = destination / "images" / subset / f"{name}{sample.image_path.suffix.lower()}"
                    dst_label = destination / "labels" / subset / f"{name}.txt"
                    transfer_file(sample.image_path, dst_image, copy_mode)
                    write_label(dst_label, lines)
                    stats[subset] += 1

    write_metadata(destination, "container_bbox")
    logging.info("Container dataset: %s", stats)
    return stats


def build_text_crop(sample: SourceSample, temp_dir: Path, expand_ratio: float) -> Optional[TextCrop]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Установите opencv-python: pip install opencv-python") from exc

    image = cv2.imread(str(sample.image_path))
    if image is None:
        logging.warning("Text crop: не удалось прочитать %s", sample.image_path)
        return None
    h, w = image.shape[:2]
    labels = read_labels(sample.label_path, w, h)
    container = choose_container(labels)
    text_boxes = [item[1:] for item in labels if item[0] == 1]
    if container is None or not text_boxes:
        return None

    x1, y1, x2, y2 = expand_box(container, w, h, expand_ratio)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop_h, crop_w = crop.shape[:2]

    lines: list[str] = []
    for tx1, ty1, tx2, ty2 in text_boxes:
        ix1 = max(float(x1), tx1)
        iy1 = max(float(y1), ty1)
        ix2 = min(float(x2), tx2)
        iy2 = min(float(y2), ty2)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        line = xyxy_to_yolo((ix1 - x1, iy1 - y1, ix2 - x1, iy2 - y1), crop_w, crop_h, 0)
        if line:
            lines.append(line)
    if not lines:
        return None

    name = safe_name(sample.number_type, sample.source, sample.image_path.stem)
    temp_image = temp_dir / f"{name}.jpg"
    temp_image.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(temp_image), crop):
        return None
    return TextCrop(sample.number_type, sample.source, temp_image, tuple(lines), name)


def create_text_datasets(
    root: Path,
    output_root: Path,
    train_ratio: float,
    expand_ratio: float,
    seed: int,
):
    summary = {}
    with tempfile.TemporaryDirectory(prefix="yolo_text_crops_") as tmp:
        temp_root = Path(tmp)

        for type_index, number_type in enumerate(TRAIN_TYPES):
            destination = output_root / "text" / number_type
            prepare_dataset_dir(destination)
            stats = {"train": 0, "val": 0, "invalid": 0}

            # Сначала создаём все валидные crops, затем делим regular/yantar отдельно.
            for source_index, source in enumerate(SOURCES):
                valid: list[TextCrop] = []
                for sample in collect_source_samples(root, number_type, source):
                    crop = build_text_crop(sample, temp_root / number_type / source, expand_ratio)
                    if crop is None:
                        stats["invalid"] += 1
                    else:
                        valid.append(crop)

                rng = random.Random(seed + 10000 + type_index * 1000 + source_index)
                train, val = split_list(valid, train_ratio, rng)

                for subset, crops in (("train", train), ("val", val)):
                    for crop in crops:
                        dst_image = destination / "images" / subset / f"{crop.name}.jpg"
                        dst_label = destination / "labels" / subset / f"{crop.name}.txt"
                        shutil.copy2(crop.image_path, dst_image)
                        write_label(dst_label, crop.label_lines)
                        stats[subset] += 1

            write_metadata(destination, "text")
            summary[number_type] = stats
            logging.info("Text dataset %s: %s", number_type, stats)

    return summary


def create_splits(
    root: Path,
    output_root: Path,
    train_ratio: float,
    expand_ratio: float,
    seed: int,
    copy_mode: str,
    keep_output: bool,
):
    if output_root.exists() and not keep_output:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    container_stats = create_container_dataset(root, output_root, train_ratio, seed, copy_mode)
    text_stats = create_text_datasets(root, output_root, train_ratio, expand_ratio, seed)
    logging.info("Split завершён: %s", output_root)
    return {"container": container_stats, "text": text_stats}


def validate_root(root: Path) -> None:
    images = root / "images"
    labels = root / "labels"
    if not images.is_dir():
        raise FileNotFoundError(f"Не найдена папка images: {images}")
    labels.mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    try:
        root = args.dataset_root.expanduser().resolve()
        output = args.output.expanduser().resolve()
        validate_root(root)

        if args.stage in ("cleanup", "all"):
            cleanup_orphan_labels(root)
        if args.stage in ("annotate", "all"):
            annotate_missing_labels(
                root,
                args.expand,
                args.container_conf,
                args.text_conf,
                args.device,
                args.overwrite,
            )
        if args.stage in ("split", "all"):
            create_splits(
                root,
                output,
                args.train_ratio,
                args.expand,
                args.seed,
                args.copy_mode,
                args.keep_output,
            )
        return 0
    except KeyboardInterrupt:
        logging.error("Операция прервана пользователем")
        return 130
    except Exception as exc:
        logging.exception("Ошибка: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
