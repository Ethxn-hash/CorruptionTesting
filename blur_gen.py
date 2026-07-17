#!/usr/bin/env python3
"""Generate normalized motion-blur corruptions for all dataset classes.

Behavior
--------
- Processes every direct class folder under --input_root.
- Selects a weighted, reproducible subset from each class:
      min(class_size, max(100, round(0.086 * class_size)))
- Generates severities 0, 10, ..., 100 for every selected image.
- Keeps generated images inside their original true-label class folder.
- Uses evaluator-compatible names such as:
      image_motion_s010_k7_a15p0.JPG
- Severity 0 is an unchanged copy.
- Severities 10-90 use normalized directional motion blur.
- Severity 100 removes all spatial information by collapsing the image
  to its mean BGR color.
- Writes _motion_blur_index.csv in the output root.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import shutil
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

DEFAULT_SEVERITIES = tuple(range(0, 101, 10))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate normalized motion-blur corruptions for every "
            "true-label class folder."
        )
    )

    parser.add_argument(
        "--input_root",
        type=Path,
        required=True,
        help="Dataset root containing one folder per true-label class.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        required=True,
        help="Root directory for generated images.",
    )
    parser.add_argument(
        "--minimum_per_class",
        type=int,
        default=100,
        help="Minimum selected images per class when available. Default: 100",
    )
    parser.add_argument(
        "--sampling_fraction",
        type=float,
        default=0.086,
        help="Sampling fraction for larger classes. Default: 0.086",
    )
    parser.add_argument(
        "--severities",
        type=str,
        default=",".join(str(value) for value in DEFAULT_SEVERITIES),
        help="Comma-separated severity levels. Default: 0,10,...,100",
    )
    parser.add_argument(
        "--angle",
        type=float,
        default=15.0,
        help="Motion-blur angle in degrees. Default: 15",
    )
    parser.add_argument(
        "--max_blur_fraction",
        type=float,
        default=0.30,
        help=(
            "Maximum kernel length for severities below 100 as a fraction "
            "of the image's smaller dimension. Default: 0.30"
        ),
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Gamma applied to normalized severity. Default: 1.0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Seed for reproducible weighted sampling. Default: 2026",
    )
    parser.add_argument(
        "--jpeg_quality",
        type=int,
        default=95,
        help="JPEG output quality. Default: 95",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite generated files that already exist.",
    )
    parser.add_argument(
        "--index_name",
        type=str,
        default="_motion_blur_index.csv",
        help="Index CSV filename.",
    )

    return parser.parse_args()


def stable_integer(text: str) -> int:
    """Return a reproducible integer derived from text."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


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


def normalized_severity(severity: float) -> float:
    """Map severity from [0, 100] to [0.0, 1.0]."""
    return float(np.clip(severity, 0.0, 100.0)) / 100.0


def nearest_odd(value: float) -> int:
    number = max(1, int(round(value)))
    return number if number % 2 == 1 else number + 1


def float_tag(value: float, decimals: int = 1) -> str:
    """Convert 15.0 to 15p0 and -15.0 to m15p0."""
    return (
        f"{value:.{decimals}f}"
        .replace("-", "m")
        .replace(".", "p")
    )


def severity_string(severity: int) -> str:
    return f"{int(severity):03d}"


def calculate_sample_count(
    class_size: int,
    minimum_per_class: int,
    sampling_fraction: float,
) -> int:
    """
    Weighted sampling rule:

        selected = min(
            class_size,
            max(minimum_per_class, round(class_size * sampling_fraction))
        )
    """
    if class_size <= 0:
        return 0

    proportional_count = round(class_size * sampling_fraction)
    requested_count = max(minimum_per_class, proportional_count)
    return min(class_size, requested_count)


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


def select_images(
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

    class_seed = seed + stable_integer(class_name)
    rng = random.Random(class_seed)

    return sorted(rng.sample(images, sample_count))


def kernel_size(
    severity: float,
    height: int,
    width: int,
    max_blur_fraction: float,
    gamma: float,
) -> int:
    """
    Normalized motion-blur scale for severities below 100:

        alpha = (severity / 100)^gamma
        maximum = nearest_odd(max_blur_fraction * min(H, W))
        size = nearest_odd(1 + (maximum - 1) * alpha)

    Severity 100 is handled separately as complete mean-color collapse.
    """
    maximum = nearest_odd(
        max_blur_fraction * min(height, width)
    )

    alpha = normalized_severity(severity) ** gamma

    return nearest_odd(
        1 + (maximum - 1) * alpha
    )


def motion_kernel(
    size: int,
    angle_degrees: float,
) -> np.ndarray:
    """Create a normalized linear motion-blur kernel."""
    if size <= 1:
        return np.array([[1.0]], dtype=np.float32)

    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    kernel[center, :] = 1.0

    rotation_matrix = cv2.getRotationMatrix2D(
        (center, center),
        angle_degrees,
        1.0,
    )

    rotated = cv2.warpAffine(
        kernel,
        rotation_matrix,
        (size, size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    total = float(rotated.sum())

    if total <= 0:
        raise RuntimeError(
            "Motion-blur kernel has zero total weight."
        )

    return rotated / total


def collapse_to_mean_color(image: np.ndarray) -> np.ndarray:
    """
    Severity-100 endpoint.

    Replace every pixel with the image's mean BGR color, removing all
    edges, lesions, texture, and spatial information.
    """
    mean_color = np.mean(
        image.astype(np.float32),
        axis=(0, 1),
        keepdims=True,
    )

    collapsed = np.broadcast_to(
        mean_color,
        image.shape,
    ).copy()

    return np.clip(
        collapsed,
        0,
        255,
    ).astype(np.uint8)


def apply_motion_corruption(
    image: np.ndarray,
    severity: int,
    angle_degrees: float,
    max_blur_fraction: float,
    gamma: float,
) -> tuple[np.ndarray | None, int, str]:
    """
    Return:
        corrupted image or None,
        kernel size,
        corruption mode

    None at severity 0 tells the writer to copy the clean source exactly.
    """
    if severity <= 0:
        return None, 1, "clean"

    if severity >= 100:
        return (
            collapse_to_mean_color(image),
            0,
            "mean_color_collapse",
        )

    height, width = image.shape[:2]

    size = kernel_size(
        severity=severity,
        height=height,
        width=width,
        max_blur_fraction=max_blur_fraction,
        gamma=gamma,
    )

    kernel = motion_kernel(
        size=size,
        angle_degrees=angle_degrees,
    )

    corrupted = cv2.filter2D(
        image,
        ddepth=-1,
        kernel=kernel,
        borderType=cv2.BORDER_REFLECT101,
    )

    return corrupted, size, "directional_motion_blur"


def read_image(path: Path) -> np.ndarray | None:
    """Read an image while supporting paths with non-ASCII characters."""
    try:
        encoded = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    except (OSError, ValueError):
        return None


def write_image(
    path: Path,
    image: np.ndarray,
    jpeg_quality: int,
) -> bool:
    """Write an image while preserving its original extension."""
    extension = path.suffix.lower()

    if extension in {".jpg", ".jpeg"}:
        encode_extension = ".jpg"
        parameters = [
            int(cv2.IMWRITE_JPEG_QUALITY),
            jpeg_quality,
        ]
    elif extension == ".png":
        encode_extension = ".png"
        parameters = [
            int(cv2.IMWRITE_PNG_COMPRESSION),
            3,
        ]
    elif extension == ".bmp":
        encode_extension = ".bmp"
        parameters = []
    elif extension in {".tif", ".tiff"}:
        encode_extension = ".tiff"
        parameters = []
    elif extension == ".webp":
        encode_extension = ".webp"
        parameters = [
            int(cv2.IMWRITE_WEBP_QUALITY),
            jpeg_quality,
        ]
    else:
        encode_extension = ".jpg"
        parameters = [
            int(cv2.IMWRITE_JPEG_QUALITY),
            jpeg_quality,
        ]

    success, encoded = cv2.imencode(
        encode_extension,
        image,
        parameters,
    )

    if not success:
        return False

    try:
        encoded.tofile(str(path))
        return True
    except OSError:
        return False


def write_or_copy(
    source_path: Path,
    output_path: Path,
    severity: int,
    corrupted: np.ndarray | None,
    overwrite: bool,
    jpeg_quality: int,
) -> str:
    """
    Return one of:
        written
        skipped
        failed
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if output_path.exists() and not overwrite:
        return "skipped"

    if severity == 0:
        try:
            shutil.copy2(source_path, output_path)
            return "written"
        except OSError:
            return "failed"

    if corrupted is None:
        raise ValueError(
            "Corrupted image cannot be None above severity 0."
        )

    if write_image(
        output_path,
        corrupted,
        jpeg_quality,
    ):
        return "written"

    return "failed"


def main() -> None:
    args = parse_args()

    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not input_root.exists():
        raise FileNotFoundError(
            f"Input root does not exist: {input_root}"
        )

    if not input_root.is_dir():
        raise NotADirectoryError(
            f"Input root is not a directory: {input_root}"
        )

    if args.minimum_per_class < 1:
        raise ValueError(
            "minimum_per_class must be at least 1."
        )

    if not 0.0 < args.sampling_fraction <= 1.0:
        raise ValueError(
            "sampling_fraction must be greater than 0 and at most 1."
        )

    if not 0.0 < args.max_blur_fraction <= 1.0:
        raise ValueError(
            "max_blur_fraction must be greater than 0 and at most 1."
        )

    if args.gamma <= 0:
        raise ValueError("gamma must be greater than zero.")

    if not 0 <= args.jpeg_quality <= 100:
        raise ValueError(
            "jpeg_quality must be between 0 and 100."
        )

    severities = parse_severities(args.severities)
    class_directories = find_class_directories(input_root)

    if not class_directories:
        raise RuntimeError(
            f"No class folders were found under: {input_root}"
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    index_path = output_root / args.index_name

    total_available = 0
    total_selected = 0
    written = 0
    skipped = 0
    failed = 0
    unreadable = 0

    print("=" * 72)
    print("Normalized Motion-Blur Generator")
    print("=" * 72)
    print(f"Input root:          {input_root}")
    print(f"Output root:         {output_root}")
    print(f"Classes found:       {len(class_directories)}")
    print(f"Minimum per class:   {args.minimum_per_class}")
    print(f"Sampling fraction:   {args.sampling_fraction:.4f}")
    print(f"Severities:          {severities}")
    print(f"Blur angle:          {args.angle}")
    print(f"Max blur fraction:   {args.max_blur_fraction}")
    print(f"Gamma:               {args.gamma}")
    print(f"Seed:                {args.seed}")
    print(f"Overwrite:           {args.overwrite}")
    print("=" * 72)

    with index_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)

        writer.writerow(
            [
                "class_name",
                "source_relative_path",
                "output_relative_path",
                "severity",
                "normalized_severity",
                "kernel_size",
                "angle_degrees",
                "corruption_mode",
                "max_blur_fraction",
                "gamma",
                "class_available_count",
                "class_selected_count",
                "sampling_fraction",
                "minimum_per_class",
                "selection_seed",
            ]
        )

        for class_directory in class_directories:
            class_name = class_directory.name
            available_images = find_images(class_directory)

            if not available_images:
                print(
                    f"[SKIP] {class_name}: no supported images"
                )
                continue

            selected_images = select_images(
                images=available_images,
                class_name=class_name,
                minimum_per_class=args.minimum_per_class,
                sampling_fraction=args.sampling_fraction,
                seed=args.seed,
            )

            total_available += len(available_images)
            total_selected += len(selected_images)

            output_class_directory = (
                output_root / class_name
            )

            output_class_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            print(
                f"[CLASS] {class_name}: "
                f"{len(selected_images)} selected from "
                f"{len(available_images)}"
            )

            for source_number, source_path in enumerate(
                selected_images,
                start=1,
            ):
                image = read_image(source_path)

                if image is None:
                    unreadable += 1
                    print(
                        f"  [READ ERROR] {source_path}"
                    )
                    continue

                source_relative_path = (
                    source_path
                    .relative_to(input_root)
                    .as_posix()
                )

                for severity in severities:
                    corrupted, size, mode = (
                        apply_motion_corruption(
                            image=image,
                            severity=severity,
                            angle_degrees=args.angle,
                            max_blur_fraction=(
                                args.max_blur_fraction
                            ),
                            gamma=args.gamma,
                        )
                    )

                    output_name = (
                        f"{source_path.stem}"
                        f"_motion_s{severity_string(severity)}"
                        f"_k{size}"
                        f"_a{float_tag(args.angle, 1)}"
                        f"{source_path.suffix}"
                    )

                    output_path = (
                        output_class_directory
                        / output_name
                    )

                    status = write_or_copy(
                        source_path=source_path,
                        output_path=output_path,
                        severity=severity,
                        corrupted=corrupted,
                        overwrite=args.overwrite,
                        jpeg_quality=args.jpeg_quality,
                    )

                    if status == "written":
                        written += 1
                    elif status == "skipped":
                        skipped += 1
                    else:
                        failed += 1
                        print(
                            f"  [WRITE ERROR] {output_path}"
                        )

                    writer.writerow(
                        [
                            class_name,
                            source_relative_path,
                            output_path
                            .relative_to(output_root)
                            .as_posix(),
                            severity,
                            (
                                f"{normalized_severity(severity):.4f}"
                            ),
                            size,
                            (
                                args.angle
                                if severity < 100
                                else ""
                            ),
                            mode,
                            args.max_blur_fraction,
                            args.gamma,
                            len(available_images),
                            len(selected_images),
                            args.sampling_fraction,
                            args.minimum_per_class,
                            args.seed,
                        ]
                    )

                if (
                    source_number % 25 == 0
                    or source_number == len(selected_images)
                ):
                    print(
                        f"  Processed "
                        f"{source_number:,}/"
                        f"{len(selected_images):,} "
                        f"source images"
                    )

    expected_outputs = (
        total_selected * len(severities)
    )

    print()
    print("=" * 72)
    print("Motion-blur generation complete")
    print("=" * 72)
    print(f"Available source images: {total_available:,}")
    print(f"Selected source images:  {total_selected:,}")
    print(f"Severity levels:         {len(severities)}")
    print(f"Expected outputs:        {expected_outputs:,}")
    print(f"Files written:           {written:,}")
    print(f"Existing files skipped:  {skipped:,}")
    print(f"Failed writes:           {failed:,}")
    print(f"Unreadable sources:      {unreadable:,}")
    print(f"Index:                   {index_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()