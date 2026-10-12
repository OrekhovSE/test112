#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import shutil
from pathlib import Path

SOURCE_TYPES = ("oneline", "twolines", "vertical")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def is_empty_label(label_path: Path) -> bool:
    """Пустой label = 0 байт или только пробелы/переводы строк/табуляция."""
    try:
        return label_path.read_text(
            encoding="utf-8-sig",
            errors="replace",
        ).strip() == ""
    except OSError as exc:
        print(f"[ERROR] Не удалось прочитать label: {label_path} -> {exc}")
        return False


def unique_pair_paths(
    image_path: Path,
    unknown_images: Path,
    unknown_labels: Path,
):
    """
    Подбирает свободное одинаковое stem-имя для картинки и label.
    Если 123.jpg уже существует, получится 123_1.jpg + 123_1.txt.
    """
    stem = image_path.stem
    image_suffix = image_path.suffix

    index = 0

    while True:
        suffix = "" if index == 0 else f"_{index}"
        candidate_stem = f"{stem}{suffix}"

        image_dst = unknown_images / f"{candidate_stem}{image_suffix}"
        label_dst = unknown_labels / f"{candidate_stem}.txt"

        if not image_dst.exists() and not label_dst.exists():
            return image_dst, label_dst

        index += 1


def move_pair(
    image_path: Path,
    label_path: Path,
    unknown_images: Path,
    unknown_labels: Path,
    dry_run: bool,
):
    image_dst, label_dst = unique_pair_paths(
        image_path=image_path,
        unknown_images=unknown_images,
        unknown_labels=unknown_labels,
    )

    print(f"[MOVE] {image_path}")
    print(f"    -> {image_dst}")
    print(f"       {label_path}")
    print(f"    -> {label_dst}")

    if dry_run:
        return

    unknown_images.mkdir(parents=True, exist_ok=True)
    unknown_labels.mkdir(parents=True, exist_ok=True)

    shutil.move(str(image_path), str(image_dst))
    shutil.move(str(label_path), str(label_dst))


def process_type(batch_root: Path, type_name: str, dry_run: bool):
    """
    Ожидаем:
      batch/images/oneline
      batch/labels/oneline
      batch/images/twolines
      batch/labels/twolines
      batch/images/vertical
      batch/labels/vertical

    Перенос:
      batch/images/unknown
      batch/labels/unknown
    """

    images_dir = batch_root / "images" / type_name
    labels_dir = batch_root / "labels" / type_name

    unknown_images = batch_root / "images" / "unknown"
    unknown_labels = batch_root / "labels" / "unknown"

    stats = {
        "batch": batch_root.name,
        "type": type_name,
        "images": 0,
        "moved": 0,
        "missing_label": 0,
        "non_empty": 0,
    }

    if not images_dir.is_dir():
        return stats

    if not labels_dir.is_dir():
        print(f"[WARN] Нет папки labels: {labels_dir}")
        return stats

    images = sorted(
        p for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    for image_path in images:
        stats["images"] += 1

        relative = image_path.relative_to(images_dir)
        label_path = (labels_dir / relative).with_suffix(".txt")

        if not label_path.exists():
            # На случай, если labels лежат без вложенных подпапок.
            flat_label = labels_dir / f"{image_path.stem}.txt"

            if flat_label.exists():
                label_path = flat_label
            else:
                stats["missing_label"] += 1
                print(f"[MISSING LABEL] {image_path}")
                continue

        if is_empty_label(label_path):
            move_pair(
                image_path=image_path,
                label_path=label_path,
                unknown_images=unknown_images,
                unknown_labels=unknown_labels,
                dry_run=dry_run,
            )
            stats["moved"] += 1
        else:
            stats["non_empty"] += 1

    return stats


def is_batch_folder(folder: Path) -> bool:
    """
    Batch считается подходящим, если есть хотя бы одна папка:
      images/oneline
      images/twolines
      images/vertical
    """
    return any(
        (folder / "images" / type_name).is_dir()
        for type_name in SOURCE_TYPES
    )


def discover_batches(root: Path):
    """
    Работает с обоими вариантами:

    Вариант 1:
      dataset/
        01.01/
          images/oneline
          labels/oneline
        02.01/
          ...

    Вариант 2:
      dataset/
        images/oneline
        labels/oneline

    В первом случае обработает все подпапки дат.
    Во втором обработает сам dataset.
    """

    if is_batch_folder(root):
        return [root]

    batches = []

    # Ищем подходящие папки по всему датасету,
    # но не заходим в images/labels как в отдельные batches.
    for folder in sorted(p for p in root.rglob("*") if p.is_dir()):
        if folder.name in {"images", "labels", "unknown", *SOURCE_TYPES}:
            continue

        if is_batch_folder(folder):
            batches.append(folder)

    return batches


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Проходит по всему датасету и переносит image+label с пустым label "
            "из images/oneline|twolines|vertical и labels/... в unknown."
        )
    )

    parser.add_argument(
        "root",
        nargs="?",
        default="dataset",
        help="Корневая папка датасета. По умолчанию: ./dataset",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Только показать, что будет перенесено, не менять файлы.",
    )

    args = parser.parse_args()

    root = Path(args.root).resolve()

    if not root.is_dir():
        raise SystemExit(f"Корневая папка не найдена: {root}")

    batches = discover_batches(root)

    if not batches:
        raise SystemExit(
            "Не найдено подходящих папок.\n"
            "Ожидается структура, например:\n"
            "dataset/01.01/images/oneline\n"
            "dataset/01.01/labels/oneline\n"
            "dataset/01.01/images/twolines\n"
            "dataset/01.01/labels/twolines\n"
            "dataset/01.01/images/vertical\n"
            "dataset/01.01/labels/vertical"
        )

    print(f"Корневая папка: {root}")
    print(f"Найдено наборов: {len(batches)}")
    print("Режим:", "DRY RUN" if args.dry_run else "MOVE")

    all_stats = []

    for batch_root in batches:
        try:
            batch_display = batch_root.relative_to(root).as_posix()
        except ValueError:
            batch_display = batch_root.name

        if batch_display == ".":
            batch_display = batch_root.name

        print(f"\n\n========== {batch_display} ==========")

        for type_name in SOURCE_TYPES:
            print(f"\n----- {type_name.upper()} -----")

            stats = process_type(
                batch_root=batch_root,
                type_name=type_name,
                dry_run=args.dry_run,
            )

            stats["batch_display"] = batch_display
            all_stats.append(stats)

    print("\n\n================ ИТОГ ================")

    total_images = 0
    total_moved = 0
    total_missing = 0
    total_non_empty = 0

    for stats in all_stats:
        total_images += stats["images"]
        total_moved += stats["moved"]
        total_missing += stats["missing_label"]
        total_non_empty += stats["non_empty"]

        if stats["images"] or stats["moved"] or stats["missing_label"]:
            print(
                f"{stats['batch_display']:<18} "
                f"{stats['type']:<10} "
                f"images={stats['images']:<6} "
                f"moved={stats['moved']:<6} "
                f"non_empty={stats['non_empty']:<6} "
                f"missing_label={stats['missing_label']:<6}"
            )

    print("--------------------------------------")
    print(f"Всего изображений проверено: {total_images}")
    print(f"С непустым label:            {total_non_empty}")
    print(f"Перенесено в unknown:        {total_moved}")
    print(f"Label не найден:             {total_missing}")

    if args.dry_run:
        print("\nЭто был dry-run. Для реального переноса запустите без --dry-run.")


if __name__ == "__main__":
    main()
