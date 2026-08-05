#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_dataset.py
==================

Скрипт подготавливает YOLO-датасет со структурой:

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

Классы:
    0 — container_bbox
    1 — text

Веса container.pt и text.pt хранятся отдельно в КАЖДОЙ соответствующей
папке labels. Например:

    images/vertical/a.jpg
        -> labels/vertical/container.pt
        -> labels/vertical/text.pt

    images/yantar/vertical/b.jpg
        -> labels/yantar/vertical/container.pt
        -> labels/yantar/vertical/text.pt

Установка зависимостей для этапа annotate:

    pip install ultralytics opencv-python

Примеры запуска
---------------

1. Удалить аннотации, для которых нет изображения:

    python prepare_dataset.py cleanup --dataset-root master_dataset

2. Создать отсутствующие аннотации:

    python prepare_dataset.py annotate --dataset-root master_dataset

3. Пересоздать все аннотации, включая существующие:

    python prepare_dataset.py annotate --dataset-root master_dataset --overwrite

4. Создать отдельные train/val датасеты по типам номеров:

    python prepare_dataset.py split \
        --dataset-root master_dataset \
        --output split_datasets \
        --train-ratio 0.8 \
        --seed 42

5. Выполнить cleanup, annotate и split последовательно:

    python prepare_dataset.py all \
        --dataset-root master_dataset \
        --output split_datasets

Дополнительные параметры:

    --expand 0.25
        Увеличить ширину и высоту container crop на 25% перед запуском text.pt.
        При значении 0.25 с каждой стороны добавляется примерно 12.5% размера bbox.

    --container-conf 0.25
        Минимальная уверенность container.pt.

    --text-conf 0.25
        Минимальная уверенность text.pt.

    --device cpu
    --device 0
        Устройство Ultralytics. Если параметр не задан, используется выбор
        Ultralytics по умолчанию.

    --copy-mode copy
    --copy-mode hardlink
        Способ формирования split. hardlink экономит место, если источник
        и результат находятся на одном диске.

Результат split
---------------

split_datasets/
├── vertical/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   ├── labels/
│   │   ├── train/
│   │   └── val/
│   ├── classes.txt
│   └── data.yaml
├── oneline/
├── twolines/
└── unknown/

Обычные данные и данные yantar делятся отдельно в пропорции 0.8/0.2,
а затем объединяются. Поэтому доля yantar получается максимально одинаковой
в train и val. Для предотвращения совпадений имён используются префиксы:

    regular__имя_файла.jpg
    yantar__имя_файла.jpg

ВАЖНО:
    - classes.txt никогда не удаляется.
    - container.pt и text.pt никогда не удаляются и не копируются в split.
    - annotate по умолчанию не изменяет существующие *.txt.
    - если container.pt отсутствует, создаётся пустой label.
    - если container найден, но text.pt отсутствует, записывается только класс 0.
    - если container не найден, создаётся пустой label.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}
NUMBER_TYPES = ("vertical", "oneline", "twolines", "unknown")
SOURCES = ("regular", "yantar")
PROTECTED_TEXT_FILES = {"classes.txt"}


@dataclass(frozen=True)
class DatasetGroup:
    number_type: str
    source: str
    image_dir: Path
    label_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Подготовка YOLO-датасета: cleanup, annotate, split или all.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "stage",
        choices=("cleanup", "annotate", "split", "all"),
        help="Этап, который необходимо выполнить.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("master_dataset"),
        help="Корень исходного датасета.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("split_datasets"),
        help="Папка результирующих split-датасетов.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Доля изображений, попадающих в train.",
    )
    parser.add_argument(
        "--expand",
        type=float,
        default=0.25,
        help="Относительное увеличение ширины/высоты crop контейнера.",
    )
    parser.add_argument(
        "--container-conf",
        type=float,
        default=0.25,
        help="Confidence threshold для container.pt.",
    )
    parser.add_argument(
        "--text-conf",
        type=float,
        default=0.25,
        help="Confidence threshold для text.pt.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Устройство Ultralytics, например cpu, 0 или 0,1.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed воспроизводимого разбиения.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Перезаписывать существующие аннотации на этапе annotate.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=("copy", "hardlink"),
        default="copy",
        help="Копировать файлы либо создавать hard links при split.",
    )
    parser.add_argument(
        "--keep-output",
        action="store_true",
        help="Не очищать существующую папку конкретного типа перед split.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Уровень логирования.",
    )
    args = parser.parse_args()

    if not 0.0 < args.train_ratio < 1.0:
        parser.error("--train-ratio должен находиться между 0 и 1.")
    if args.expand < 0.0:
        parser.error("--expand не может быть отрицательным.")
    if not 0.0 <= args.container_conf <= 1.0:
        parser.error("--container-conf должен находиться между 0 и 1.")
    if not 0.0 <= args.text_conf <= 1.0:
        parser.error("--text-conf должен находиться между 0 и 1.")

    return args


def iter_groups(dataset_root: Path) -> Iterable[DatasetGroup]:
    images_root = dataset_root / "images"
    labels_root = dataset_root / "labels"

    for number_type in NUMBER_TYPES:
        yield DatasetGroup(
            number_type=number_type,
            source="regular",
            image_dir=images_root / number_type,
            label_dir=labels_root / number_type,
        )
        yield DatasetGroup(
            number_type=number_type,
            source="yantar",
            image_dir=images_root / "yantar" / number_type,
            label_dir=labels_root / "yantar" / number_type,
        )


def list_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_stems(directory: Path) -> set[str]:
    return {path.stem for path in list_images(directory)}


def cleanup_orphan_labels(dataset_root: Path) -> int:
    removed = 0

    for group in iter_groups(dataset_root):
        if not group.label_dir.is_dir():
            continue

        stems = image_stems(group.image_dir)
        for label_path in sorted(group.label_dir.glob("*.txt")):
            if label_path.name.lower() in PROTECTED_TEXT_FILES:
                continue
            if label_path.stem not in stems:
                label_path.unlink()
                removed += 1
                logging.info("Удалён label без изображения: %s", label_path)

    logging.info("Cleanup завершён. Удалено labels: %d", removed)
    return removed


class ModelCache:
    def __init__(self) -> None:
        self._models: dict[Path, object] = {}

    def get(self, weights_path: Path):
        resolved = weights_path.resolve()
        if resolved not in self._models:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError(
                    "Для annotate установите ultralytics: pip install ultralytics"
                ) from exc
            logging.info("Загрузка весов: %s", weights_path)
            self._models[resolved] = YOLO(str(weights_path))
        return self._models[resolved]


def predict_xyxy(
    model,
    image,
    confidence: float,
    device: Optional[str],
) -> list[tuple[float, float, float, float, float]]:
    kwargs = {
        "source": image,
        "conf": confidence,
        "verbose": False,
    }
    if device is not None:
        kwargs["device"] = device

    results = model.predict(**kwargs)
    detections: list[tuple[float, float, float, float, float]] = []

    if not results:
        return detections

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return detections

    xyxy_values = boxes.xyxy.detach().cpu().tolist()
    conf_values = boxes.conf.detach().cpu().tolist()

    for coords, conf in zip(xyxy_values, conf_values):
        x1, y1, x2, y2 = map(float, coords)
        detections.append((x1, y1, x2, y2, float(conf)))

    return detections


def clip_box(
    box: Sequence[float],
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    return x1, y1, x2, y2


def expand_box(
    box: Sequence[float],
    width: int,
    height: int,
    expand_ratio: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = clip_box(box, width, height)
    box_width = max(0.0, x2 - x1)
    box_height = max(0.0, y2 - y1)

    pad_x = box_width * expand_ratio / 2.0
    pad_y = box_height * expand_ratio / 2.0

    expanded = clip_box(
        (x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y),
        width,
        height,
    )
    ex1, ey1, ex2, ey2 = expanded
    return (
        int(ex1),
        int(ey1),
        int(ex2 + 0.999999),
        int(ey2 + 0.999999),
    )


def xyxy_to_yolo(
    box: Sequence[float],
    image_width: int,
    image_height: int,
    class_id: int,
) -> Optional[str]:
    x1, y1, x2, y2 = clip_box(box, image_width, image_height)
    box_width = x2 - x1
    box_height = y2 - y1

    if box_width <= 0.0 or box_height <= 0.0:
        return None

    center_x = (x1 + x2) / 2.0 / image_width
    center_y = (y1 + y2) / 2.0 / image_height
    norm_width = box_width / image_width
    norm_height = box_height / image_height

    values = (center_x, center_y, norm_width, norm_height)
    values = tuple(max(0.0, min(1.0, value)) for value in values)

    return (
        f"{class_id} "
        f"{values[0]:.6f} {values[1]:.6f} "
        f"{values[2]:.6f} {values[3]:.6f}"
    )


def write_label(label_path: Path, lines: Sequence[str]) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines)
    if content:
        content += "\n"
    label_path.write_text(content, encoding="utf-8")


def annotate_missing_labels(
    dataset_root: Path,
    expand_ratio: float,
    container_conf: float,
    text_conf: float,
    device: Optional[str],
    overwrite: bool,
) -> dict[str, int]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "Для annotate установите OpenCV: pip install opencv-python"
        ) from exc

    cache = ModelCache()
    stats = {
        "processed": 0,
        "skipped_existing": 0,
        "empty_no_container_weights": 0,
        "empty_no_detection": 0,
        "container_only": 0,
        "container_and_text": 0,
        "read_errors": 0,
    }

    for group in iter_groups(dataset_root):
        images = list_images(group.image_dir)
        if not images:
            continue

        group.label_dir.mkdir(parents=True, exist_ok=True)
        container_weights = group.label_dir / "container.pt"
        text_weights = group.label_dir / "text.pt"

        for image_path in images:
            label_path = group.label_dir / f"{image_path.stem}.txt"

            if label_path.exists() and not overwrite:
                stats["skipped_existing"] += 1
                continue

            stats["processed"] += 1

            if not container_weights.is_file():
                write_label(label_path, [])
                stats["empty_no_container_weights"] += 1
                logging.warning(
                    "Нет %s; создан пустой label: %s",
                    container_weights,
                    label_path,
                )
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                write_label(label_path, [])
                stats["read_errors"] += 1
                logging.error("Не удалось прочитать изображение: %s", image_path)
                continue

            image_height, image_width = image.shape[:2]
            container_model = cache.get(container_weights)
            container_detections = predict_xyxy(
                container_model,
                image,
                container_conf,
                device,
            )

            if not container_detections:
                write_label(label_path, [])
                stats["empty_no_detection"] += 1
                logging.info("Контейнер не найден: %s", image_path)
                continue

            # Для container_bbox выбирается детекция с максимальной уверенностью.
            best_container = max(container_detections, key=lambda item: item[4])
            container_box = best_container[:4]

            lines: list[str] = []
            container_line = xyxy_to_yolo(
                container_box,
                image_width,
                image_height,
                class_id=0,
            )
            if container_line is not None:
                lines.append(container_line)

            if not text_weights.is_file():
                write_label(label_path, lines)
                stats["container_only"] += 1
                logging.warning(
                    "Нет %s; записан только container_bbox: %s",
                    text_weights,
                    label_path,
                )
                continue

            crop_x1, crop_y1, crop_x2, crop_y2 = expand_box(
                container_box,
                image_width,
                image_height,
                expand_ratio,
            )
            crop = image[crop_y1:crop_y2, crop_x1:crop_x2]

            if crop.size == 0:
                write_label(label_path, lines)
                stats["container_only"] += 1
                logging.warning("Получен пустой crop: %s", image_path)
                continue

            text_model = cache.get(text_weights)
            text_detections = predict_xyxy(
                text_model,
                crop,
                text_conf,
                device,
            )

            # Стабильный порядок: сверху вниз, затем слева направо.
            text_detections.sort(key=lambda item: (item[1], item[0]))

            for tx1, ty1, tx2, ty2, _confidence in text_detections:
                source_box = (
                    tx1 + crop_x1,
                    ty1 + crop_y1,
                    tx2 + crop_x1,
                    ty2 + crop_y1,
                )
                text_line = xyxy_to_yolo(
                    source_box,
                    image_width,
                    image_height,
                    class_id=1,
                )
                if text_line is not None:
                    lines.append(text_line)

            write_label(label_path, lines)
            if len(lines) > 1:
                stats["container_and_text"] += 1
            else:
                stats["container_only"] += 1

            logging.info(
                "Создан label: %s (container=1, text=%d)",
                label_path,
                max(0, len(lines) - 1),
            )

    logging.info("Annotate завершён: %s", stats)
    return stats


def split_group(
    items: Sequence[Path],
    train_ratio: float,
    rng: random.Random,
) -> tuple[list[Path], list[Path]]:
    shuffled = list(items)
    rng.shuffle(shuffled)
    count = len(shuffled)

    if count == 0:
        return [], []
    if count == 1:
        return shuffled, []

    train_count = round(count * train_ratio)
    train_count = max(1, min(count - 1, train_count))

    return shuffled[:train_count], shuffled[train_count:]


def transfer_file(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        destination.unlink()

    if mode == "hardlink":
        try:
            os.link(source, destination)
            return
        except OSError as exc:
            logging.warning(
                "Hardlink не создан (%s), используется копирование: %s",
                exc,
                source,
            )

    shutil.copy2(source, destination)


def copy_split_item(
    image_path: Path,
    source_label_dir: Path,
    destination_root: Path,
    subset: str,
    source_name: str,
    copy_mode: str,
) -> None:
    prefix = "regular__" if source_name == "regular" else "yantar__"
    output_stem = f"{prefix}{image_path.stem}"

    output_image = (
        destination_root / "images" / subset / f"{output_stem}{image_path.suffix.lower()}"
    )
    output_label = destination_root / "labels" / subset / f"{output_stem}.txt"
    source_label = source_label_dir / f"{image_path.stem}.txt"

    transfer_file(image_path, output_image, copy_mode)

    if source_label.is_file():
        transfer_file(source_label, output_label, copy_mode)
    else:
        # split может запускаться отдельно от annotate.
        write_label(output_label, [])
        logging.warning(
            "Для изображения нет label; в split создан пустой: %s",
            source_label,
        )


def find_classes_file(dataset_root: Path, number_type: str) -> Optional[Path]:
    candidates = (
        dataset_root / "labels" / number_type / "classes.txt",
        dataset_root / "labels" / "yantar" / number_type / "classes.txt",
        dataset_root / "labels" / "classes.txt",
    )
    return next((path for path in candidates if path.is_file()), None)


def write_data_yaml(destination_root: Path) -> None:
    yaml_content = """path: .
train: images/train
val: images/val

names:
  0: container_bbox
  1: text
"""
    (destination_root / "data.yaml").write_text(yaml_content, encoding="utf-8")


def create_splits(
    dataset_root: Path,
    output_root: Path,
    train_ratio: float,
    seed: int,
    copy_mode: str,
    keep_output: bool,
) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}

    for type_index, number_type in enumerate(NUMBER_TYPES):
        destination_root = output_root / number_type

        if destination_root.exists() and not keep_output:
            shutil.rmtree(destination_root)

        for subset in ("train", "val"):
            (destination_root / "images" / subset).mkdir(parents=True, exist_ok=True)
            (destination_root / "labels" / subset).mkdir(parents=True, exist_ok=True)

        counts = {
            "regular_train": 0,
            "regular_val": 0,
            "yantar_train": 0,
            "yantar_val": 0,
        }

        for source_index, source_name in enumerate(SOURCES):
            if source_name == "regular":
                image_dir = dataset_root / "images" / number_type
                label_dir = dataset_root / "labels" / number_type
            else:
                image_dir = dataset_root / "images" / "yantar" / number_type
                label_dir = dataset_root / "labels" / "yantar" / number_type

            images = list_images(image_dir)
            # Разные детерминированные RNG для каждого типа и источника.
            rng = random.Random(seed + type_index * 1000 + source_index)
            train_items, val_items = split_group(images, train_ratio, rng)

            for subset, subset_items in (
                ("train", train_items),
                ("val", val_items),
            ):
                for image_path in subset_items:
                    copy_split_item(
                        image_path=image_path,
                        source_label_dir=label_dir,
                        destination_root=destination_root,
                        subset=subset,
                        source_name=source_name,
                        copy_mode=copy_mode,
                    )
                counts[f"{source_name}_{subset}"] = len(subset_items)

        classes_file = find_classes_file(dataset_root, number_type)
        if classes_file is not None:
            shutil.copy2(classes_file, destination_root / "classes.txt")
        else:
            (destination_root / "classes.txt").write_text(
                "container_bbox\ntext\n",
                encoding="utf-8",
            )

        write_data_yaml(destination_root)
        summary[number_type] = counts
        logging.info("Split %s: %s", number_type, counts)

    logging.info("Split завершён. Результат: %s", output_root)
    return summary


def validate_dataset_root(dataset_root: Path) -> None:
    images_root = dataset_root / "images"
    labels_root = dataset_root / "labels"

    if not images_root.is_dir():
        raise FileNotFoundError(f"Не найдена папка images: {images_root}")
    if not labels_root.exists():
        labels_root.mkdir(parents=True, exist_ok=True)
        logging.warning("Создана отсутствующая папка labels: %s", labels_root)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    try:
        dataset_root = args.dataset_root.expanduser().resolve()
        output_root = args.output.expanduser().resolve()
        validate_dataset_root(dataset_root)

        if args.stage in ("cleanup", "all"):
            cleanup_orphan_labels(dataset_root)

        if args.stage in ("annotate", "all"):
            annotate_missing_labels(
                dataset_root=dataset_root,
                expand_ratio=args.expand,
                container_conf=args.container_conf,
                text_conf=args.text_conf,
                device=args.device,
                overwrite=args.overwrite,
            )

        if args.stage in ("split", "all"):
            create_splits(
                dataset_root=dataset_root,
                output_root=output_root,
                train_ratio=args.train_ratio,
                seed=args.seed,
                copy_mode=args.copy_mode,
                keep_output=args.keep_output,
            )

        return 0

    except KeyboardInterrupt:
        logging.error("Операция прервана пользователем.")
        return 130
    except Exception as exc:
        logging.exception("Ошибка: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
