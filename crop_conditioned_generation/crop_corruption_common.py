#!/usr/bin/env python3
"""Shared utilities for configurable crop-only corruption generation."""

from __future__ import annotations

import csv
import hashlib
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import cv2
import numpy as np


IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}


@dataclass(frozen=True)
class SelectedImage:
    clean_root: Path
    crop: str
    class_name: str
    relative_path: Path
    original_class_count: int
    selected_class_count: int

    @property
    def source_path(self) -> Path:
        return self.clean_root / self.relative_path


def severity_string(value: int | float) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return str(number).replace(".", "p")


def float_tag(value: float, decimals: int = 4) -> str:
    return f"{value:.{decimals}f}".replace(".", "p")


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def is_inside(child: Path, parent: Path) -> bool:
    child = child.resolve()
    parent = parent.resolve()
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def validate_crop_groups(groups: Mapping[str, Sequence[str]]) -> Dict[str, str]:
    if not groups:
        raise ValueError("CROP_CLASS_GROUPS cannot be empty.")

    class_to_crop: Dict[str, str] = {}
    for crop, class_names in groups.items():
        if not class_names:
            raise ValueError(f"Crop {crop!r} has no class names.")
        if len(class_names) < 4:
            print(
                f"WARNING: crop {crop!r} has only {len(class_names)} classes; "
                "the intended experiment requested crops with at least four."
            )
        for class_name in class_names:
            if class_name in class_to_crop:
                raise ValueError(
                    f"Class {class_name!r} appears in more than one crop group."
                )
            class_to_crop[class_name] = crop
    return class_to_crop


def validate_severities(values: Iterable[int | float]) -> List[int]:
    severities: List[int] = []
    for value in values:
        number = int(value)
        if float(value) != number:
            raise ValueError("SEVERITIES must contain integer values.")
        if number < 0 or number > 100:
            raise ValueError(f"Severity must be in [0, 100], got {number}.")
        if number not in severities:
            severities.append(number)
    if not severities:
        raise ValueError("SEVERITIES cannot be empty.")
    return severities


def stable_seed(global_seed: int, class_name: str) -> int:
    digest = hashlib.sha256(
        f"{global_seed}:{class_name}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def selected_count(available: int, minimum: int, rate: float) -> int:
    if available <= 0:
        return 0
    return min(available, max(minimum, int(round(rate * available))))


def discover_target_images(
    clean_root: Path,
    crop_groups: Mapping[str, Sequence[str]],
) -> Dict[str, List[Path]]:
    clean_root = clean_root.expanduser().resolve()
    if not clean_root.is_dir():
        raise NotADirectoryError(f"Clean dataset folder not found: {clean_root}")

    class_to_crop = validate_crop_groups(crop_groups)
    missing = [
        class_name
        for class_name in class_to_crop
        if not (clean_root / class_name).is_dir()
    ]
    if missing:
        raise FileNotFoundError(
            "These configured class folders were not found under CLEAN_DATASET:\n- "
            + "\n- ".join(missing)
        )

    discovered: Dict[str, List[Path]] = {}
    for class_name in class_to_crop:
        class_dir = clean_root / class_name
        images = sorted(
            path.relative_to(clean_root)
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            raise RuntimeError(f"No supported images found in: {class_dir}")
        discovered[class_name] = images
    return discovered


def create_manifest(
    clean_root: Path,
    manifest_path: Path,
    crop_groups: Mapping[str, Sequence[str]],
    minimum: int,
    rate: float,
    seed: int,
) -> List[SelectedImage]:
    if minimum < 1:
        raise ValueError("MIN_PER_CLASS must be at least 1.")
    if not 0 < rate <= 1:
        raise ValueError("SAMPLE_RATE must be in (0, 1].")

    clean_root = clean_root.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    class_to_crop = validate_crop_groups(crop_groups)
    images_by_class = discover_target_images(clean_root, crop_groups)

    selected: List[SelectedImage] = []
    for class_name in sorted(images_by_class):
        available_paths = images_by_class[class_name]
        count = selected_count(len(available_paths), minimum, rate)
        rng = random.Random(stable_seed(seed, class_name))
        chosen = sorted(rng.sample(available_paths, count))
        for relative_path in chosen:
            selected.append(
                SelectedImage(
                    clean_root=clean_root,
                    crop=class_to_crop[class_name],
                    class_name=class_name,
                    relative_path=relative_path,
                    original_class_count=len(available_paths),
                    selected_class_count=count,
                )
            )

    selected.sort(key=lambda row: row.relative_path.as_posix())
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "clean_root",
                "crop",
                "class_name",
                "relative_path",
                "original_class_count",
                "selected_class_count",
                "random_seed",
                "min_per_class",
                "sample_rate",
            ],
        )
        writer.writeheader()
        for row in selected:
            writer.writerow(
                {
                    "clean_root": str(clean_root),
                    "crop": row.crop,
                    "class_name": row.class_name,
                    "relative_path": row.relative_path.as_posix(),
                    "original_class_count": row.original_class_count,
                    "selected_class_count": row.selected_class_count,
                    "random_seed": seed,
                    "min_per_class": minimum,
                    "sample_rate": rate,
                }
            )

    summary_path = manifest_path.with_name(manifest_path.stem + "_summary.csv")
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["crop", "class_name", "available_images", "selected_images"]
        )
        for class_name in sorted(images_by_class):
            rows = [row for row in selected if row.class_name == class_name]
            writer.writerow(
                [
                    class_to_crop[class_name],
                    class_name,
                    len(images_by_class[class_name]),
                    len(rows),
                ]
            )

    return selected


def load_manifest(
    clean_root: Path,
    manifest_path: Path,
    crop_groups: Mapping[str, Sequence[str]],
) -> List[SelectedImage]:
    clean_root = clean_root.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    class_to_crop = validate_crop_groups(crop_groups)
    rows: List[SelectedImage] = []
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "crop",
            "class_name",
            "relative_path",
            "original_class_count",
            "selected_class_count",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "Manifest is missing columns: " + ", ".join(sorted(missing))
            )

        seen = set()
        for item in reader:
            class_name = item["class_name"]
            if class_name not in class_to_crop:
                raise ValueError(
                    f"Manifest contains a class not in CROP_CLASS_GROUPS: "
                    f"{class_name}"
                )
            crop = item["crop"]
            if crop != class_to_crop[class_name]:
                raise ValueError(
                    f"Manifest crop mismatch for {class_name}: {crop!r}"
                )
            relative_path = Path(item["relative_path"])
            if relative_path.as_posix() in seen:
                raise ValueError(
                    f"Duplicate relative path in manifest: {relative_path}"
                )
            seen.add(relative_path.as_posix())
            source_path = clean_root / relative_path
            if not source_path.is_file():
                raise FileNotFoundError(
                    f"Manifest source image no longer exists: {source_path}"
                )
            rows.append(
                SelectedImage(
                    clean_root=clean_root,
                    crop=crop,
                    class_name=class_name,
                    relative_path=relative_path,
                    original_class_count=int(item["original_class_count"]),
                    selected_class_count=int(item["selected_class_count"]),
                )
            )

    rows.sort(key=lambda row: row.relative_path.as_posix())
    if not rows:
        raise RuntimeError("Manifest contains no selected images.")
    return rows


def prepare_generation(config, factor: str):
    clean_root = Path(config.CLEAN_DATASET).expanduser().resolve()
    output_attr = {
        "motion_blur": "MOTION_BLUR_OUTPUT",
        "resolution": "RESOLUTION_OUTPUT",
        "lighting": "LIGHTING_OUTPUT",
    }[factor]
    output_root = Path(getattr(config, output_attr)).expanduser().resolve()
    manifest_path = Path(config.MANIFEST_PATH).expanduser().resolve()

    if is_inside(output_root, clean_root):
        raise ValueError(
            f"{output_attr} cannot be inside CLEAN_DATASET; otherwise generated "
            "images may be rediscovered as clean inputs."
        )

    severities = validate_severities(config.SEVERITIES)
    if config.RECREATE_MANIFEST or not manifest_path.exists():
        selected = create_manifest(
            clean_root=clean_root,
            manifest_path=manifest_path,
            crop_groups=config.CROP_CLASS_GROUPS,
            minimum=int(config.MIN_PER_CLASS),
            rate=float(config.SAMPLE_RATE),
            seed=int(config.RANDOM_SEED),
        )
        print(f"Created manifest: {manifest_path}")
    else:
        selected = load_manifest(
            clean_root=clean_root,
            manifest_path=manifest_path,
            crop_groups=config.CROP_CLASS_GROUPS,
        )
        print(f"Loaded manifest: {manifest_path}")

    output_root.mkdir(parents=True, exist_ok=True)
    print(f"Factor: {factor}")
    print(f"Selected clean images: {len(selected):,}")
    print(f"Severity levels: {severities}")
    print(f"Expected output images: {len(selected) * len(severities):,}")
    print(f"Output folder: {output_root}\n")
    return clean_root, output_root, severities, selected


def build_output_path(
    output_root: Path,
    source_relative_path: Path,
    output_name: str,
) -> Path:
    output_path = output_root / source_relative_path.parent / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def save_image(path: Path, image: np.ndarray, jpeg_quality: int = 95) -> None:
    params: List[int] = []
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
    if not cv2.imwrite(str(path), image, params):
        raise IOError(f"OpenCV failed to save: {path}")


def write_or_copy(
    source_path: Path,
    output_path: Path,
    severity: int,
    corrupted: np.ndarray | None,
    overwrite: bool,
    jpeg_quality: int,
) -> str:
    if output_path.exists() and not overwrite:
        return "skipped"
    if severity == 0:
        shutil.copy2(source_path, output_path)
    else:
        if corrupted is None:
            raise ValueError("A corrupted image is required for severity > 0.")
        save_image(output_path, corrupted, jpeg_quality)
    return "written"
