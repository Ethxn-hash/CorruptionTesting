#!/usr/bin/env python3
"""Generate sampled PlantVillage lighting corruptions for all classes.

Default behavior exactly matches corruption_gen_test/lighting.py and its
normalized severity tables.

The prior standalone lighting controls and filename/index structure are
retained. Passing --legacy-spatial-formula uses the previous configurable
spatial multiplier formula; without that flag, the repository-matched
luminance-based corruption is used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import shutil
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


VALID_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp",
}

LIGHTING_PUSH_TABLE = {
    0: 0.00,
    10: 0.10,
    20: 0.20,
    30: 0.30,
    40: 0.40,
    50: 0.50,
    60: 0.60,
    70: 0.70,
    80: 0.80,
    90: 0.90,
    100: 1.00,
}

LIGHTING_BW_BLEND_TABLE = {
    0: 0.00,
    10: 0.00,
    20: 0.00,
    30: 0.02,
    40: 0.06,
    50: 0.12,
    60: 0.22,
    70: 0.36,
    80: 0.54,
    90: 0.76,
    100: 1.00,
}


@dataclass
class SelectedImage:
    crop: str
    class_name: str
    source_path: Path
    relative_path: Path


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def severity_string(value: float) -> str:
    return f"{int(round(value)):03d}"


def float_tag(value: float, decimals: int = 2) -> str:
    return (
        f"{value:.{decimals}f}"
        .replace("-", "m")
        .replace(".", "p")
    )


def stable_integer(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def infer_crop(class_name: str) -> str:
    if "___" in class_name:
        return class_name.split("___", 1)[0]
    if "__" in class_name:
        return class_name.split("__", 1)[0]
    return class_name


def parse_severities(text: str) -> list[int]:
    severities = sorted({
        int(part.strip())
        for part in text.split(",")
        if part.strip()
    })

    if not severities:
        raise ValueError("At least one severity must be provided.")

    for severity in severities:
        if not 0 <= severity <= 100:
            raise ValueError(
                f"Severity {severity} is outside the valid range 0-100."
            )

    return severities


def interpolate_table(table: dict[int, float], severity: float) -> float:
    severity = clamp(float(severity), 0.0, 100.0)
    keys = sorted(table)

    if severity <= keys[0]:
        return float(table[keys[0]])

    if severity >= keys[-1]:
        return float(table[keys[-1]])

    exact = int(severity)
    if float(severity).is_integer() and exact in table:
        return float(table[exact])

    upper_index = bisect_right(keys, severity)
    lower_key = keys[upper_index - 1]
    upper_key = keys[upper_index]

    fraction = (
        (severity - lower_key)
        / (upper_key - lower_key)
    )

    return (
        float(table[lower_key])
        + fraction
        * (
            float(table[upper_key])
            - float(table[lower_key])
        )
    )


def get_lighting_push(severity: float) -> float:
    return clamp(
        interpolate_table(LIGHTING_PUSH_TABLE, severity),
        0.0,
        1.0,
    )


def get_lighting_bw_blend(severity: float) -> float:
    return clamp(
        interpolate_table(LIGHTING_BW_BLEND_TABLE, severity),
        0.0,
        1.0,
    )


def calculate_sample_count(
    class_size: int,
    minimum_per_class: int,
    sampling_fraction: float,
) -> int:
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

    output_root.mkdir(parents=True, exist_ok=True)
    severities = parse_severities(severities_text)
    class_directories = find_class_directories(input_root)

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
                    relative_path=image_path.relative_to(input_root),
                )
            )

    if not selected:
        raise RuntimeError(
            "No images were selected for generation."
        )

    return input_root, output_root, severities, selected


def build_output_path(
    output_root: Path,
    relative_path: Path,
    output_name: str,
) -> Path:
    output_directory = output_root / relative_path.parent
    output_directory.mkdir(parents=True, exist_ok=True)
    return output_directory / output_name


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if image.ndim == 3:
        if image.shape[2] == 1:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if image.shape[2] == 3:
            return image

    raise ValueError(
        f"Unsupported image shape: {image.shape}"
    )


def force_odd(value: float) -> int:
    value = max(3, int(round(value)))
    if value % 2 == 0:
        value += 1
    return value


def create_repo_lighting_map(
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    image = ensure_bgr(image)
    height, width = image.shape[:2]
    minimum_dimension = min(height, width)

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB,
    )

    luminance = (
        lab[:, :, 0].astype(np.float32)
        / 255.0
    )

    sigma = float(
        np.clip(
            minimum_dimension * 0.035,
            3.0,
            35.0,
        )
    )

    kernel_size = force_odd(
        6.0 * sigma + 1.0
    )

    smoothed = cv2.GaussianBlur(
        luminance,
        (kernel_size, kernel_size),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT101,
    )

    low = float(np.percentile(smoothed, 2))
    high = float(np.percentile(smoothed, 98))

    if high - low < 1e-6:
        normalized_map = np.full_like(
            smoothed,
            0.5,
            dtype=np.float32,
        )
    else:
        normalized_map = (
            (smoothed - low)
            / (high - low)
        )
        normalized_map = np.clip(
            normalized_map,
            0.0,
            1.0,
        ).astype(np.float32)

    signed_map = (
        2.0 * normalized_map - 1.0
    ).astype(np.float32)

    return normalized_map, signed_map


def apply_repo_lighting(
    image: np.ndarray,
    severity: float,
) -> tuple[np.ndarray, float, float]:
    image = ensure_bgr(image)
    severity = clamp(float(severity), 0.0, 100.0)

    if severity <= 0:
        return image.copy(), 0.0, 0.0

    push = clamp(
        get_lighting_push(severity),
        0.0,
        1.5,
    )

    bw_blend = clamp(
        get_lighting_bw_blend(severity),
        0.0,
        1.0,
    )

    image_float = image.astype(np.float32)
    normalized_map, signed_map = (
        create_repo_lighting_map(image)
    )

    region_strength = (
        np.abs(signed_map) ** 0.55
    ).astype(np.float32)

    signed_map_3 = signed_map[:, :, None]
    strength_3 = region_strength[:, :, None]

    bright_weight = (
        np.clip(signed_map_3, 0.0, 1.0)
        * strength_3
    )

    dark_weight = (
        np.clip(-signed_map_3, 0.0, 1.0)
        * strength_3
    )

    bright_change = (
        push
        * bright_weight
        * (255.0 - image_float)
    )

    dark_change = (
        push
        * dark_weight
        * image_float
    )

    illumination_image = np.clip(
        image_float
        + bright_change
        - dark_change,
        0.0,
        255.0,
    )

    binary_mask = normalized_map >= 0.5

    binary_gray = np.where(
        binary_mask,
        255.0,
        0.0,
    ).astype(np.float32)

    binary_target = np.repeat(
        binary_gray[:, :, None],
        3,
        axis=2,
    )

    corrupted = (
        (1.0 - bw_blend) * illumination_image
        + bw_blend * binary_target
    )

    if severity >= 100:
        corrupted = binary_target.copy()

    corrupted = np.clip(
        corrupted,
        0.0,
        255.0,
    ).astype(np.uint8)

    return ensure_bgr(corrupted), push, bw_blend


def spatial_pattern(
    height: int,
    width: int,
    pattern: str,
) -> np.ndarray:
    x = np.linspace(0, 1, width, dtype=np.float32)
    y = np.linspace(0, 1, height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    if pattern == "diagonal":
        values = 0.5 * (xx + yy)
    elif pattern == "horizontal":
        values = xx
    elif pattern == "vertical":
        values = yy
    elif pattern in {"radial", "vignette"}:
        distance = np.sqrt(
            (xx - 0.5) ** 2
            + (yy - 0.5) ** 2
        )
        maximum = np.sqrt(0.5**2 + 0.5**2)
        values = 1.0 - distance / maximum

        if pattern == "vignette":
            values = 1.0 - values
    else:
        raise ValueError(
            "pattern must be diagonal, horizontal, vertical, "
            "radial, or vignette."
        )

    return np.clip(values, 0, 1).astype(np.float32)


def apply_legacy_lighting(
    image: np.ndarray,
    severity: float,
    minimum_multiplier: float,
    maximum_multiplier: float,
    sharpness: float,
    pattern_name: str,
) -> np.ndarray:
    height, width = image.shape[:2]
    alpha = clamp(severity, 0, 100) / 100.0
    pattern = spatial_pattern(
        height,
        width,
        pattern_name,
    )

    split = 1.0 / (
        1.0
        + np.exp(
            -sharpness * (pattern - 0.5)
        )
    )

    target = (
        minimum_multiplier
        + (
            maximum_multiplier
            - minimum_multiplier
        )
        * split
    )

    multiplier = (
        (1.0 - alpha)
        + alpha * target
    )

    if image.ndim == 3:
        multiplier = multiplier[:, :, None]

    result = image.astype(np.float32) * multiplier

    return np.clip(
        result,
        0,
        255,
    ).astype(np.uint8)


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
            [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
        )
    elif extension == ".png":
        success, encoded = cv2.imencode(
            ".png",
            image,
            [int(cv2.IMWRITE_PNG_COMPRESSION), 3],
        )
    elif extension == ".bmp":
        success, encoded = cv2.imencode(".bmp", image)
    elif extension in {".tif", ".tiff"}:
        success, encoded = cv2.imencode(".tiff", image)
    elif extension == ".webp":
        success, encoded = cv2.imencode(
            ".webp",
            image,
            [int(cv2.IMWRITE_WEBP_QUALITY), jpeg_quality],
        )
    else:
        success, encoded = cv2.imencode(
            ".jpg",
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
        )

    if not success:
        raise IOError(f"OpenCV failed to encode: {path}")

    encoded.tofile(str(path))


def write_or_copy(
    source_path: Path,
    output_path: Path,
    severity: float,
    corrupted: np.ndarray | None,
    overwrite: bool,
    jpeg_quality: int,
) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        return "skipped"

    if severity == 0:
        shutil.copy2(source_path, output_path)
        return "written"

    if corrupted is None:
        raise ValueError(
            "Corrupted image cannot be None above severity 0."
        )

    save_cv_image(output_path, corrupted, jpeg_quality)
    return "written"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a sampled 11-level PlantVillage lighting dataset "
            "using corruption_gen_test severity by default."
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
        help="Root directory for generated lighting images.",
    )
    parser.add_argument(
        "--severities",
        type=str,
        default="0,10,20,30,40,50,60,70,80,90,100",
        help="Comma-separated severity levels.",
    )
    parser.add_argument(
        "--minimum-per-class",
        type=int,
        default=100,
        help="Minimum selected images per class when available.",
    )
    parser.add_argument(
        "--sampling-fraction",
        type=float,
        default=0.086,
        help="Weighted class sampling fraction.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--minimum-multiplier",
        type=float,
        default=0.0,
        help=(
            "Retained for the optional legacy spatial formula. "
            "Ignored by the default repository-matched formula."
        ),
    )
    parser.add_argument(
        "--maximum-multiplier",
        type=float,
        default=6.0,
        help=(
            "Retained for the optional legacy spatial formula. "
            "Ignored by the default repository-matched formula."
        ),
    )
    parser.add_argument(
        "--sharpness",
        type=float,
        default=12.0,
        help=(
            "Retained for the optional legacy spatial formula. "
            "Ignored by the default repository-matched formula."
        ),
    )
    parser.add_argument(
        "--pattern",
        choices=[
            "diagonal",
            "horizontal",
            "vertical",
            "radial",
            "vignette",
        ],
        default="diagonal",
        help=(
            "Retained for the optional legacy spatial formula. "
            "Ignored by the default repository-matched formula."
        ),
    )
    parser.add_argument(
        "--legacy-spatial-formula",
        action="store_true",
        help=(
            "Use the previous configurable spatial multiplier formula "
            "instead of the default repository-matched severity."
        ),
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality.",
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
        help="Print progress every N source images.",
    )

    args = parser.parse_args()

    if args.maximum_multiplier < args.minimum_multiplier:
        raise ValueError(
            "maximum_multiplier must be >= minimum_multiplier."
        )

    if args.sharpness <= 0:
        raise ValueError(
            "sharpness must be positive."
        )

    if not 0 <= args.jpeg_quality <= 100:
        raise ValueError(
            "jpeg_quality must be between 0 and 100."
        )

    _, output_root, severities, selected = prepare_run(
        input_root=args.input_root,
        output_root=args.output_root,
        severities_text=args.severities,
        minimum_per_class=args.minimum_per_class,
        sampling_fraction=args.sampling_fraction,
        seed=args.seed,
    )

    index_path = output_root / "_lighting_index.csv"
    written = skipped = unreadable = 0

    min_tag = float_tag(args.minimum_multiplier, 2)
    max_tag = float_tag(args.maximum_multiplier, 2)
    sharp_tag = float_tag(args.sharpness, 2)

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
                "minimum_multiplier",
                "maximum_multiplier",
                "sharpness",
                "pattern",
            ]
        )

        for number, row in enumerate(selected, start=1):
            image = cv2.imread(
                str(row.source_path),
                cv2.IMREAD_COLOR,
            )

            if image is None:
                print(
                    f"WARNING: unreadable image skipped: {row.source_path}"
                )
                unreadable += 1
                continue

            for severity in severities:
                output_name = (
                    f"{row.source_path.stem}"
                    f"_lighting_s{severity_string(severity)}"
                    f"_min{min_tag}"
                    f"_max{max_tag}"
                    f"_sh{sharp_tag}"
                    f"_{args.pattern}"
                    f"{row.source_path.suffix}"
                )

                output_path = build_output_path(
                    output_root,
                    row.relative_path,
                    output_name,
                )

                corrupted = None

                if severity > 0:
                    if args.legacy_spatial_formula:
                        corrupted = apply_legacy_lighting(
                            image=image,
                            severity=severity,
                            minimum_multiplier=args.minimum_multiplier,
                            maximum_multiplier=args.maximum_multiplier,
                            sharpness=args.sharpness,
                            pattern_name=args.pattern,
                        )
                    else:
                        corrupted, _, _ = apply_repo_lighting(
                            image,
                            severity,
                        )

                status = write_or_copy(
                    row.source_path,
                    output_path,
                    severity,
                    corrupted,
                    args.overwrite,
                    args.jpeg_quality,
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
                        args.minimum_multiplier,
                        args.maximum_multiplier,
                        args.sharpness,
                        args.pattern,
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
                    f"Processed {number:,}/{len(selected):,} "
                    f"source images."
                )

    print("\nLighting generation complete.")
    print(f"Files written: {written:,}")
    print(f"Existing files skipped: {skipped:,}")
    print(f"Unreadable source images: {unreadable:,}")
    print(f"Index: {index_path}")


if __name__ == "__main__":
    main()