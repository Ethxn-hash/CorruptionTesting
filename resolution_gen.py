#!/usr/bin/env python3
"""Generate normalized resolution-degraded images for all class folders.

This standalone version preserves the original generator behavior while
removing the dependency on generation_config.py and crop_corruption_common.py.

Normalized severity:
    normalized_severity = clamp(severity, 0, 100) / 100

Resolution scale:
    scale = 1 - (1 - minimum_scale) * normalized_severity

Therefore:
    severity 0   -> scale 1.0, exact clean copy
    severity 100 -> scale minimum_scale

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
    crop: str
    class_name: str
    source_path: Path
    relative_path: Path


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalized_severity(severity: float) -> float:
    """Convert severity from [0, 100] to [0.0, 1.0]."""
    return clamp(severity, 0.0, 100.0) / 100.0


def severity_string(value: float) -> str:
    return f"{int(round(value)):03d}"


def float_tag(value: float, decimals: int = 4) -> str:
    return (
        f"{value:.{decimals}f}"
        .replace("-", "m")
        .replace(".", "p")
    )


def stable_integer(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def infer_crop(class_name: str) -> str:
    """
    Infer the crop name from PlantVillage-style class folders.

    Examples:
        Apple___Apple_scab -> Apple
        Corn__Common_rust  -> Corn
        Tomato___healthy   -> Tomato
    """
    if "___" in class_name:
        return class_name.split("___", 1)[0]

    if "__" in class_name:
        return class_name.split("__", 1)[0]

    return class_name


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
    """
    Weighted class sampling rule:

        selected_count = min(
            class_size,
            max(
                minimum_per_class,
                round(class_size * sampling_fraction)
            )
        )
    """
    if class_size <= 0:
        return 0

    return min(
        class_size,
        max(
            minimum_per_class,
            round(class_size * sampling_fraction),
        ),
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
        if path.is_file()
        and path.suffix.lower() in VALID_EXTENSIONS
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

    rng = random.Random(
        seed + stable_integer(class_name)
    )

    return sorted(
        rng.sample(images, sample_count)
    )


def prepare_run(
    input_root: Path,
    output_root: Path,
    severities_text: str,
    minimum_per_class: int,
    sampling_fraction: float,
    seed: int,
) -> tuple[Path, Path, list[int], list[SelectedImage]]:
    input_root = input_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()

    if not input_root.exists():
        raise FileNotFoundError(
            f"Input root does not exist: {input_root}"
        )

    if not input_root.is_dir():
        raise NotADirectoryError(
            f"Input root is not a directory: {input_root}"
        )

    if minimum_per_class < 1:
        raise ValueError(
            "minimum_per_class must be at least 1."
        )

    if not 0.0 < sampling_fraction <= 1.0:
        raise ValueError(
            "sampling_fraction must be greater than 0 and at most 1."
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    severities = parse_severities(
        severities_text
    )

    class_directories = find_class_directories(
        input_root
    )

    if not class_directories:
        raise RuntimeError(
            f"No class directories found under: {input_root}"
        )

    selected: list[SelectedImage] = []

    for class_directory in class_directories:
        class_name = class_directory.name
        crop = infer_crop(class_name)
        images = find_images(class_directory)

        if not images:
            print(
                f"WARNING: no supported images in {class_directory}"
            )
            continue

        chosen = select_images_for_class(
            images=images,
            class_name=class_name,
            minimum_per_class=minimum_per_class,
            sampling_fraction=sampling_fraction,
            seed=seed,
        )

        print(
            f"[CLASS] {class_name}: "
            f"{len(chosen)} selected from {len(images)}"
        )

        for image_path in chosen:
            selected.append(
                SelectedImage(
                    crop=crop,
                    class_name=class_name,
                    source_path=image_path,
                    relative_path=(
                        image_path.relative_to(input_root)
                    ),
                )
            )

    if not selected:
        raise RuntimeError(
            "No images were selected for generation."
        )

    return (
        input_root,
        output_root,
        severities,
        selected,
    )


def build_output_path(
    output_root: Path,
    relative_path: Path,
    output_name: str,
) -> Path:
    output_directory = (
        output_root / relative_path.parent
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_directory / output_name


def scale_factor(
    severity: float,
    minimum_scale: float,
) -> float:
    """
    Normalized resolution mapping:

        n = clamp(severity, 0, 100) / 100
        scale = 1 - (1 - minimum_scale) * n
    """
    if not 0.0 < minimum_scale <= 1.0:
        raise ValueError(
            "minimum_scale must be in (0, 1]."
        )

    norm_severity = normalized_severity(
        severity
    )

    scale = (
        1.0
        - (1.0 - minimum_scale)
        * norm_severity
    )

    return clamp(
        scale,
        minimum_scale,
        1.0,
    )


def degrade(
    image: np.ndarray,
    severity: float,
    minimum_scale: float,
    upsample_method: str,
):
    height, width = image.shape[:2]

    scale = scale_factor(
        severity=severity,
        minimum_scale=minimum_scale,
    )

    small_width = max(
        1,
        int(round(width * scale)),
    )

    small_height = max(
        1,
        int(round(height * scale)),
    )

    reduced = cv2.resize(
        image,
        (small_width, small_height),
        interpolation=cv2.INTER_AREA,
    )

    methods = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
    }

    method_name = upsample_method.lower()

    if method_name not in methods:
        raise ValueError(
            "upsample_method must be nearest, linear, or cubic."
        )

    restored = cv2.resize(
        reduced,
        (width, height),
        interpolation=methods[method_name],
    )

    return (
        restored,
        scale,
        small_width,
        small_height,
    )


def save_cv_image(
    path: Path,
    image: np.ndarray,
    jpeg_quality: int,
) -> None:
    extension = path.suffix.lower()

    if extension in {".jpg", ".jpeg"}:
        success, encoded = cv2.imencode(
            ".jpg",
            image,
            [
                int(cv2.IMWRITE_JPEG_QUALITY),
                jpeg_quality,
            ],
        )

    elif extension == ".png":
        success, encoded = cv2.imencode(
            ".png",
            image,
            [
                int(cv2.IMWRITE_PNG_COMPRESSION),
                3,
            ],
        )

    elif extension == ".bmp":
        success, encoded = cv2.imencode(
            ".bmp",
            image,
        )

    elif extension in {".tif", ".tiff"}:
        success, encoded = cv2.imencode(
            ".tiff",
            image,
        )

    elif extension == ".webp":
        success, encoded = cv2.imencode(
            ".webp",
            image,
            [
                int(cv2.IMWRITE_WEBP_QUALITY),
                jpeg_quality,
            ],
        )

    else:
        success, encoded = cv2.imencode(
            ".jpg",
            image,
            [
                int(cv2.IMWRITE_JPEG_QUALITY),
                jpeg_quality,
            ],
        )

    if not success:
        raise IOError(
            f"OpenCV failed to encode: {path}"
        )

    try:
        encoded.tofile(str(path))
    except OSError as error:
        raise IOError(
            f"Failed to save: {path}"
        ) from error


def write_or_copy(
    source_path: Path,
    output_path: Path,
    severity: float,
    corrupted: np.ndarray | None,
    overwrite: bool,
    jpeg_quality: int,
) -> str:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if output_path.exists() and not overwrite:
        return "skipped"

    if severity == 0:
        shutil.copy2(
            source_path,
            output_path,
        )
        return "written"

    if corrupted is None:
        raise ValueError(
            "Corrupted image cannot be None above severity 0."
        )

    save_cv_image(
        output_path,
        corrupted,
        jpeg_quality,
    )

    return "written"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a sampled 11-level PlantVillage resolution dataset "
            "while preserving the clean class-folder structure."
        )
    )

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
        help="Root directory for generated resolution images.",
    )

    parser.add_argument(
        "--severities",
        type=str,
        default=(
            "0,10,20,30,40,50,"
            "60,70,80,90,100"
        ),
        help=(
            "Comma-separated severity levels. "
            "Default: 0,10,...,100"
        ),
    )

    parser.add_argument(
        "--minimum-per-class",
        type=int,
        default=100,
        help=(
            "Minimum selected images per class when available. "
            "Default: 100"
        ),
    )

    parser.add_argument(
        "--sampling-fraction",
        type=float,
        default=0.086,
        help=(
            "Weighted class sampling fraction. "
            "Default: 0.086"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help=(
            "Random seed for reproducible sampling. "
            "Default: 2026"
        ),
    )

    parser.add_argument(
        "--minimum-scale",
        type=float,
        default=0.03,
        help=(
            "Retained width and height scale at severity 100. "
            "Default: 0.03"
        ),
    )

    parser.add_argument(
        "--upsample-method",
        choices=[
            "nearest",
            "linear",
            "cubic",
        ],
        default="linear",
        help=(
            "Upsampling interpolation. "
            "Default: linear"
        ),
    )

    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality. Default: 95",
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
        help=(
            "Print progress every N source images. "
            "Default: 25"
        ),
    )

    args = parser.parse_args()

    if not 0.0 < args.minimum_scale <= 1.0:
        raise ValueError(
            "minimum_scale must be in (0, 1]."
        )

    if not 0 <= args.jpeg_quality <= 100:
        raise ValueError(
            "jpeg_quality must be between 0 and 100."
        )

    (
        _,
        output_root,
        severities,
        selected,
    ) = prepare_run(
        input_root=args.input_root,
        output_root=args.output_root,
        severities_text=args.severities,
        minimum_per_class=args.minimum_per_class,
        sampling_fraction=args.sampling_fraction,
        seed=args.seed,
    )

    index_path = (
        output_root / "_resolution_index.csv"
    )

    written = 0
    skipped = 0
    unreadable = 0

    with index_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)

        writer.writerow(
            [
                "crop",
                "class_name",
                "source_relative_path",
                "output_relative_path",
                "severity",
                "scale_factor",
                "downsample_width",
                "downsample_height",
                "upsample_method",
                "minimum_scale",
            ]
        )

        for number, row in enumerate(
            selected,
            start=1,
        ):
            image = cv2.imread(
                str(row.source_path),
                cv2.IMREAD_COLOR,
            )

            if image is None:
                print(
                    "WARNING: unreadable image skipped: "
                    f"{row.source_path}"
                )
                unreadable += 1
                continue

            for severity in severities:
                height, width = image.shape[:2]

                scale = scale_factor(
                    severity=severity,
                    minimum_scale=args.minimum_scale,
                )

                small_width = max(
                    1,
                    int(round(width * scale)),
                )

                small_height = max(
                    1,
                    int(round(height * scale)),
                )

                output_name = (
                    f"{row.source_path.stem}"
                    f"_resolution_s"
                    f"{severity_string(severity)}"
                    f"_Sc{float_tag(scale, 4)}"
                    f"{row.source_path.suffix}"
                )

                output_path = build_output_path(
                    output_root,
                    row.relative_path,
                    output_name,
                )

                corrupted = None

                if severity > 0:
                    (
                        corrupted,
                        scale,
                        small_width,
                        small_height,
                    ) = degrade(
                        image=image,
                        severity=severity,
                        minimum_scale=args.minimum_scale,
                        upsample_method=args.upsample_method,
                    )

                status = write_or_copy(
                    source_path=row.source_path,
                    output_path=output_path,
                    severity=severity,
                    corrupted=corrupted,
                    overwrite=args.overwrite,
                    jpeg_quality=args.jpeg_quality,
                )

                written += status == "written"
                skipped += status == "skipped"

                writer.writerow(
                    [
                        row.crop,
                        row.class_name,
                        row.relative_path.as_posix(),
                        output_path
                        .relative_to(output_root)
                        .as_posix(),
                        severity,
                        f"{scale:.8f}",
                        small_width,
                        small_height,
                        args.upsample_method,
                        args.minimum_scale,
                    ]
                )

            if (
                args.progress_every > 0
                and (
                    number % args.progress_every == 0
                    or number == len(selected)
                )
            ):
                print(
                    f"Processed {number:,}/"
                    f"{len(selected):,} "
                    f"source images."
                )

    print("\nResolution generation complete.")
    print(f"Files written: {written:,}")
    print(
        f"Existing files skipped: {skipped:,}"
    )
    print(
        f"Unreadable source images: {unreadable:,}"
    )
    print(f"Index: {index_path}")


if __name__ == "__main__":
    main()