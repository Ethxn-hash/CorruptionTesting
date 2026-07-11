#!/usr/bin/env python3
"""
Shared utilities for deterministic PlantVillage corruption generation.

Expected clean-dataset layout:
    CLEAN_ROOT/
        Class_A/
            image1.jpg
            image2.jpg
        Class_B/
            ...

The first directory below CLEAN_ROOT is treated as the class name. Images may
also be nested more deeply inside each class; the full relative structure is
preserved in the output.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}

DEFAULT_SEVERITIES = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
DEFAULT_MIN_PER_CLASS = 100
DEFAULT_SAMPLE_RATE = 0.086
DEFAULT_SEED = 2026


@dataclass(frozen=True)
class SelectedImage:
    dataset_root: Path
    class_name: str
    relative_path: Path
    original_class_count: int
    selected_class_count: int
    global_seed: int

    @property
    def source_path(self) -> Path:
        return self.dataset_root / self.relative_path


def path_is_inside(child: Path, parent: Path) -> bool:
    """Return True when child is equal to or contained by parent."""
    child = child.resolve()
    parent = parent.resolve()
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def parse_severities(text: Optional[str]) -> List[int]:
    if text is None or not text.strip():
        return list(DEFAULT_SEVERITIES)

    values: List[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value < 0 or value > 100:
            raise ValueError(f"Severity must be in [0, 100], got {value}.")
        values.append(value)

    if not values:
        raise ValueError("At least one severity is required.")

    # Preserve the user's order while removing duplicates.
    return list(dict.fromkeys(values))


def severity_string(severity: int | float) -> str:
    value = float(severity)
    if value.is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def float_to_filename(value: float, decimals: int = 4) -> str:
    return f"{value:.{decimals}f}".replace(".", "p")


def discover_class_images(input_root: Path) -> Dict[str, List[Path]]:
    """
    Return images grouped by the first folder below input_root.

    Returned paths are relative to input_root.
    """
    input_root = input_root.resolve()
    if not input_root.is_dir():
        raise NotADirectoryError(f"Clean dataset folder not found: {input_root}")

    grouped: Dict[str, List[Path]] = {}

    for class_dir in sorted(p for p in input_root.iterdir() if p.is_dir()):
        relative_images = sorted(
            p.relative_to(input_root)
            for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if relative_images:
            grouped[class_dir.name] = relative_images

    if not grouped:
        raise RuntimeError(
            "No class folders containing supported images were found. "
            "Point --input at the folder whose immediate subfolders are classes."
        )

    return grouped


def stable_class_seed(global_seed: int, class_name: str) -> int:
    """
    Give every class an independent deterministic seed.

    This prevents a change in one class from changing the selections in all
    later classes.
    """
    token = f"{global_seed}:{class_name}".encode("utf-8")
    digest = hashlib.sha256(token).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def selected_count_for_class(
    available_count: int,
    min_per_class: int,
    sample_rate: float,
) -> int:
    """
    Hybrid allocation:
        selected = min(available, max(min_per_class, round(rate * available)))

    With the standard 54,305-image, 38-class PlantVillage distribution and the
    defaults min_per_class=100 and sample_rate=0.086, this selects 5,450 unique
    source images.
    """
    if available_count < 1:
        return 0
    proportional = int(round(sample_rate * available_count))
    return min(available_count, max(min_per_class, proportional))


def create_manifest(
    input_root: Path,
    manifest_path: Path,
    min_per_class: int = DEFAULT_MIN_PER_CLASS,
    sample_rate: float = DEFAULT_SAMPLE_RATE,
    seed: int = DEFAULT_SEED,
) -> List[SelectedImage]:
    if min_per_class < 1:
        raise ValueError("--min-per-class must be at least 1.")
    if not 0 < sample_rate <= 1:
        raise ValueError("--sample-rate must be in the interval (0, 1].")

    input_root = input_root.resolve()
    class_images = discover_class_images(input_root)

    selected_rows: List[SelectedImage] = []

    for class_name, relative_images in class_images.items():
        available = len(relative_images)
        selected_count = selected_count_for_class(
            available_count=available,
            min_per_class=min_per_class,
            sample_rate=sample_rate,
        )

        rng = random.Random(stable_class_seed(seed, class_name))
        selected_paths = sorted(rng.sample(relative_images, selected_count))

        selected_rows.extend(
            SelectedImage(
                dataset_root=input_root,
                class_name=class_name,
                relative_path=relative_path,
                original_class_count=available,
                selected_class_count=selected_count,
                global_seed=seed,
            )
            for relative_path in selected_paths
        )

    selected_rows.sort(key=lambda row: row.relative_path.as_posix())

    manifest_path = manifest_path.resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset_root",
                "class_name",
                "relative_path",
                "original_class_count",
                "selected_class_count",
                "global_seed",
                "min_per_class",
                "sample_rate",
            ],
        )
        writer.writeheader()
        for row in selected_rows:
            writer.writerow(
                {
                    "dataset_root": str(row.dataset_root),
                    "class_name": row.class_name,
                    "relative_path": row.relative_path.as_posix(),
                    "original_class_count": row.original_class_count,
                    "selected_class_count": row.selected_class_count,
                    "global_seed": row.global_seed,
                    "min_per_class": min_per_class,
                    "sample_rate": sample_rate,
                }
            )

    write_sampling_summary(selected_rows, manifest_path.with_name(
        manifest_path.stem + "_summary.csv"
    ))

    return selected_rows


def load_manifest(input_root: Path, manifest_path: Path) -> List[SelectedImage]:
    input_root = input_root.resolve()
    manifest_path = manifest_path.resolve()

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    rows: List[SelectedImage] = []
    seen_paths = set()

    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "dataset_root",
            "class_name",
            "relative_path",
            "original_class_count",
            "selected_class_count",
            "global_seed",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Manifest is missing required column(s): {sorted(missing)}"
            )

        for record in reader:
            manifest_root = Path(record["dataset_root"]).resolve()
            if manifest_root != input_root:
                raise ValueError(
                    "The existing manifest belongs to a different clean dataset.\n"
                    f"Manifest dataset: {manifest_root}\n"
                    f"Current --input:  {input_root}\n"
                    "Use the correct manifest or pass --rebuild-manifest."
                )

            relative_path = Path(record["relative_path"])
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(
                    f"Unsafe relative path in manifest: {relative_path}"
                )

            if relative_path in seen_paths:
                raise ValueError(
                    f"Duplicate image in manifest: {relative_path.as_posix()}"
                )
            seen_paths.add(relative_path)

            source_path = input_root / relative_path
            if not source_path.is_file():
                raise FileNotFoundError(
                    f"Manifest source image no longer exists: {source_path}"
                )

            rows.append(
                SelectedImage(
                    dataset_root=input_root,
                    class_name=record["class_name"],
                    relative_path=relative_path,
                    original_class_count=int(record["original_class_count"]),
                    selected_class_count=int(record["selected_class_count"]),
                    global_seed=int(record["global_seed"]),
                )
            )

    if not rows:
        raise ValueError(f"Manifest contains no selected images: {manifest_path}")

    rows.sort(key=lambda row: row.relative_path.as_posix())
    return rows


def write_sampling_summary(
    selected_rows: Sequence[SelectedImage],
    output_path: Path,
) -> None:
    summary: Dict[str, Tuple[int, int]] = {}
    for row in selected_rows:
        summary[row.class_name] = (
            row.original_class_count,
            row.selected_class_count,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["class_name", "original_images", "selected_source_images"]
        )
        for class_name in sorted(summary):
            original_count, selected_count = summary[class_name]
            writer.writerow([class_name, original_count, selected_count])
        writer.writerow(
            [
                "TOTAL",
                sum(original for original, _ in summary.values()),
                sum(selected for _, selected in summary.values()),
            ]
        )


def load_or_create_manifest(args: argparse.Namespace) -> List[SelectedImage]:
    input_root = Path(args.input).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()

    if args.rebuild_manifest and manifest_path.exists():
        manifest_path.unlink()
        summary_path = manifest_path.with_name(manifest_path.stem + "_summary.csv")
        if summary_path.exists():
            summary_path.unlink()

    if manifest_path.exists():
        selected = load_manifest(input_root, manifest_path)
        print(f"Loaded shared selection manifest: {manifest_path}")
    else:
        selected = create_manifest(
            input_root=input_root,
            manifest_path=manifest_path,
            min_per_class=args.min_per_class,
            sample_rate=args.sample_rate,
            seed=args.seed,
        )
        print(f"Created shared selection manifest: {manifest_path}")

    return selected


def build_output_path(
    output_root: Path,
    source_relative_path: Path,
    output_name: str,
) -> Path:
    """
    Preserve all source subfolders while replacing only the filename.
    """
    destination = output_root / source_relative_path.parent / output_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Clean dataset root. Its immediate subfolders must be the classes."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output root for this corruption factor.",
    )
    parser.add_argument(
        "--manifest",
        default="plantvillage_selected_images.csv",
        help=(
            "Shared CSV selection manifest. Use the same path for all three "
            "factor programs. Created automatically if it does not exist."
        ),
    )
    parser.add_argument(
        "--severities",
        default="0,10,20,30,40,50,60,70,80,90,100",
        help="Comma-separated severity levels. Default: 0 through 100 by 10.",
    )
    parser.add_argument(
        "--min-per-class",
        type=int,
        default=DEFAULT_MIN_PER_CLASS,
        help="Minimum unique source images per class. Default: 100.",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=DEFAULT_SAMPLE_RATE,
        help=(
            "Proportional allocation for larger classes. Default: 0.086. "
            "Combined with a 100-image floor, this selects 5,450 images from "
            "the standard PlantVillage dataset."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Deterministic sampling seed. Default: 2026.",
    )
    parser.add_argument(
        "--rebuild-manifest",
        action="store_true",
        help=(
            "Delete and recreate the shared manifest using the current "
            "sampling options. Do this only before running the first factor."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output images that already exist. Default: skip them.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress after this many source images. Default: 100.",
    )


def prepare_run(
    args: argparse.Namespace,
    factor_label: str,
) -> Tuple[Path, Path, List[int], List[SelectedImage]]:
    input_root = Path(args.input).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()

    if not input_root.is_dir():
        raise NotADirectoryError(f"Input dataset not found: {input_root}")

    if path_is_inside(output_root, input_root):
        raise ValueError(
            "The output folder cannot be inside the clean input dataset. "
            "That could cause generated images to be sampled as clean images."
        )

    output_root.mkdir(parents=True, exist_ok=True)
    severities = parse_severities(args.severities)
    selected = load_or_create_manifest(args)

    class_count = len({row.class_name for row in selected})
    print(f"Factor: {factor_label}")
    print(f"Clean dataset: {input_root}")
    print(f"Output dataset: {output_root}")
    print(f"Classes represented: {class_count}")
    print(f"Unique selected source images: {len(selected):,}")
    print(f"Severity levels: {severities}")
    print(
        f"Expected image files for this factor: "
        f"{len(selected) * len(severities):,}"
    )

    return input_root, output_root, severities, selected


def write_index_header(writer: csv.writer, extra_columns: Iterable[str]) -> None:
    writer.writerow(
        [
            "class_name",
            "source_relative_path",
            "output_relative_path",
            "severity",
            *extra_columns,
        ]
    )
