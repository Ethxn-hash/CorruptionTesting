#!/usr/bin/env python3
"""
Generate the sampled PlantVillage motion-blur dataset.

Standalone version of the repository-style generator.

Formula used for severities 1-99:
    L_max = nearest_odd(max_blur_fraction * min(H, W))
    k(S)  = nearest_odd(1 + (L_max - 1) * (S / 100)^gamma)

Corrected normalized severity endpoint:
    severity 0   -> exact clean copy
    severity 1-99 -> directional motion blur
    severity 100 -> complete information collapse to mean BGR color

Default severities: 0, 10, ..., 100.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


@dataclass
class SelectedImage:
    class_name: str
    source_path: Path
    relative_path: Path


def nearest_odd(value: float) -> int:
    number = max(1, int(round(value)))
    if number % 2 == 0:
        number += 1
    return number


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def stable_integer(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def severity_string(value: float) -> str:
    return f"{int(round(value)):03d}"


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Root directory containing one folder per class.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Root directory where generated images will be saved.",
    )
    parser.add_argument(
        "--severities",
        type=str,
        default="0,10,20,30,40,50,60,70,80,90,100",
        help="Comma-separated severity levels. Default: 0,10,...,100",
    )
    parser.add_argument(
        "--minimum-per-class",
        type=int,
        default=100,
        help="Minimum sampled images per class when available. Default: 100",
    )
    parser.add_argument(
        "--sampling-fraction",
        type=float,
        default=0.086,
        help="Weighted class sampling fraction. Default: 0.086",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed for reproducible sampling. Default: 2026",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing generated files.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print progress every N source images. Default: 25",
    )


def parse_severities(text: str) -> list[int]:
    severities = sorted(
        {
            int(part.strip())
            for part in text.split(",")
            if part.strip()
        }
    )

    if not severities:
        raise ValueError("At least one severity must be provided.")

    for severity in severities:
        if not 0 <= severity <= 100:
            raise ValueError(
                f"Severity {severity} is outside the valid range 0-100."
            )

    return severities


def calculate_sample_count(
    class_size: int,
    minimum_per_class: int,
    sampling_fraction: float,
) -> int:
    return min(
        class_size,
        max(minimum_per_class, round(class_size * sampling_fraction)),
    )


def find_class_directories(input_root: Path) -> list[Path]:
    return sorted(
        path
        for path in input_root.iterdir()
        if path.is_dir()
    )


def find_images(class_directory: Path) -> list[Path]:
    return sorted(
        path
        for path in class_directory.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    )


def select_images_for_class(
    images: list[Path],
    class_name: str,
    minimum_per_class: int,
    sampling_fraction: float,
    seed: int,
) -> list[Path]:
    sample_count = calculate_sample_count(
        class_size=len(images),
        minimum_per_class=minimum_per_class,
        sampling_fraction=sampling_fraction,
    )

    if sample_count >= len(images):
        return list(images)

    rng = random.Random(seed + stable_integer(class_name))
    return sorted(rng.sample(images, sample_count))


def prepare_run(
    args: argparse.Namespace,
    factor_label: str,
) -> tuple[Path, Path, list[int], list[SelectedImage]]:
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")
    if not input_root.is_dir():
        raise NotADirectoryError(f"Input root is not a directory: {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)

    severities = parse_severities(args.severities)
    class_directories = find_class_directories(input_root)

    if not class_directories:
        raise RuntimeError(f"No class directories found under: {input_root}")

    selected: list[SelectedImage] = []

    for class_directory in class_directories:
        class_name = class_directory.name
        images = find_images(class_directory)

        if not images:
            continue

        chosen = select_images_for_class(
            images=images,
            class_name=class_name,
            minimum_per_class=args.minimum_per_class,
            sampling_fraction=args.sampling_fraction,
            seed=args.seed,
        )

        for image_path in chosen:
            selected.append(
                SelectedImage(
                    class_name=class_name,
                    source_path=image_path,
                    relative_path=image_path.relative_to(input_root),
                )
            )

    if not selected:
        raise RuntimeError("No images were selected for generation.")

    return input_root, output_root, severities, selected


def build_output_path(
    output_root: Path,
    relative_path: Path,
    output_name: str,
) -> Path:
    class_folder = relative_path.parent
    output_directory = output_root / class_folder
    output_directory.mkdir(parents=True, exist_ok=True)
    return output_directory / output_name


def write_index_header(writer: csv.writer, extra_columns: list[str]) -> None:
    writer.writerow(
        [
            "class_name",
            "source_relative_path",
            "output_relative_path",
            "severity",
            *extra_columns,
        ]
    )


def motion_kernel_size(
    severity: float,
    height: int,
    width: int,
    max_blur_fraction: float = 0.30,
    gamma: float = 2.0,
) -> int:
    severity = clamp(severity, 0, 100)
    minimum_dimension = min(height, width)
    maximum_length = nearest_odd(max_blur_fraction * minimum_dimension)
    return nearest_odd(
        1
        + (maximum_length - 1)
        * ((severity / 100.0) ** gamma)
    )


def create_motion_blur_kernel(
    kernel_size: int,
    angle_degrees: float,
) -> np.ndarray:
    if kernel_size <= 1:
        return np.array([[1.0]], dtype=np.float32)

    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2
    kernel[center, :] = 1.0

    rotation_matrix = cv2.getRotationMatrix2D(
        (center, center),
        angle_degrees,
        1.0,
    )
    rotated_kernel = cv2.warpAffine(
        kernel,
        rotation_matrix,
        (kernel_size, kernel_size),
    )

    kernel_sum = float(rotated_kernel.sum())
    if kernel_sum != 0:
        rotated_kernel /= kernel_sum

    return rotated_kernel


def collapse_to_mean_color(image: np.ndarray) -> np.ndarray:
    mean_color = np.mean(
        image.astype(np.float32),
        axis=(0, 1),
        keepdims=True,
    )
    collapsed = np.broadcast_to(mean_color, image.shape).copy()
    return np.clip(collapsed, 0, 255).astype(np.uint8)


def apply_motion_blur(
    image: np.ndarray,
    severity: float,
    angle_degrees: float = 0.0,
    max_blur_fraction: float = 0.30,
    gamma: float = 2.0,
) -> Tuple[np.ndarray, int]:
    """
    Corrected normalized-severity behavior:

    severity 0:
        clean image (handled outside this function in main)

    severity 1-99:
        directional motion blur with repository-style formula

    severity 100:
        complete mean-color collapse
    """
    severity = clamp(severity, 0, 100)

    if severity >= 100:
        collapsed = collapse_to_mean_color(image)
        return collapsed, 0

    height, width = image.shape[:2]
    kernel_size = motion_kernel_size(
        severity=severity,
        height=height,
        width=width,
        max_blur_fraction=max_blur_fraction,
        gamma=gamma,
    )
    kernel = create_motion_blur_kernel(kernel_size, angle_degrees)
    blurred = cv2.filter2D(image, ddepth=-1, kernel=kernel)
    return blurred, kernel_size


def save_cv_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise IOError(f"OpenCV failed to save: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a sampled 11-level PlantVillage motion-blur dataset while "
            "preserving the clean folder structure."
        )
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--angle",
        type=float,
        default=0.0,
        help="Motion direction in degrees. Default: 0.",
    )
    parser.add_argument(
        "--max-blur-fraction",
        type=float,
        default=0.30,
        help=(
            "Severity-100 blur length as a fraction of min(H, W) for the "
            "blurred portion of the scale. Default: 0.30."
        ),
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=2.0,
        help="Severity-curve exponent. Default: 2.0.",
    )
    args = parser.parse_args()

    _, output_root, severities, selected = prepare_run(
        args,
        factor_label="motion_blur",
    )

    index_path = output_root / "_motion_blur_index.csv"
    generated = 0
    skipped_existing = 0
    unreadable = 0

    with index_path.open("w", newline="", encoding="utf-8") as index_handle:
        writer = csv.writer(index_handle)
        write_index_header(
            writer,
            ["kernel_size", "angle_degrees", "max_blur_fraction", "gamma"],
        )

        for source_number, row in enumerate(selected, start=1):
            source_path = row.source_path
            image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)

            if image is None:
                print(f"WARNING: unreadable image skipped: {source_path}")
                unreadable += 1
                continue

            height, width = image.shape[:2]

            for severity in severities:
                if severity >= 100:
                    kernel_size = 0
                else:
                    kernel_size = motion_kernel_size(
                        severity=severity,
                        height=height,
                        width=width,
                        max_blur_fraction=args.max_blur_fraction,
                        gamma=args.gamma,
                    )

                output_name = (
                    f"{source_path.stem}"
                    f"_motion_s{severity_string(severity)}"
                    f"_k{kernel_size}"
                    f"_a{severity_string(args.angle)}"
                    f"{source_path.suffix}"
                )
                output_path = build_output_path(
                    output_root,
                    row.relative_path,
                    output_name,
                )

                if output_path.exists() and not args.overwrite:
                    skipped_existing += 1
                elif severity == 0:
                    # Preserve the clean file exactly; avoid JPEG recompression.
                    shutil.copy2(source_path, output_path)
                    generated += 1
                else:
                    corrupted, _ = apply_motion_blur(
                        image=image,
                        severity=severity,
                        angle_degrees=args.angle,
                        max_blur_fraction=args.max_blur_fraction,
                        gamma=args.gamma,
                    )
                    save_cv_image(output_path, corrupted)
                    generated += 1

                writer.writerow(
                    [
                        row.class_name,
                        row.relative_path.as_posix(),
                        output_path.relative_to(output_root).as_posix(),
                        severity,
                        kernel_size,
                        args.angle,
                        args.max_blur_fraction,
                        args.gamma,
                    ]
                )

            if (
                args.progress_every > 0
                and (
                    source_number % args.progress_every == 0
                    or source_number == len(selected)
                )
            ):
                print(
                    f"Processed {source_number:,}/{len(selected):,} "
                    f"source images."
                )

    print("\nMotion-blur generation complete.")
    print(f"New files written: {generated:,}")
    print(f"Existing files skipped: {skipped_existing:,}")
    print(f"Unreadable source images: {unreadable:,}")
    print(f"Output index: {index_path}")


if __name__ == "__main__":
    main()