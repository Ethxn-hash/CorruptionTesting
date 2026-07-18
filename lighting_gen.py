#!/usr/bin/env python3
"""Generate the full sampled lighting-corruption dataset.

Standalone version matching the other dataset generators:
- processes every direct class folder under --input-root
- weighted reproducible sampling per class
- at least 100 images per class when available
- severities 0, 10, ..., 100 by default
- preserves true-label class folders
- severity 0 is copied exactly
- writes _lighting_index.csv
- requires no configuration/helper file

The lighting formula and values are copied from the approved preview script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


# =====================================================================
# LOCKED LIGHTING SETTINGS FROM THE APPROVED PREVIEW
# =====================================================================
PATTERN = "diagonal"
SEVERITY_GAMMA = 2.00
HIGHLIGHT_STOPS = 7.00
SHADOW_STOPS = 8.00
FIELD_CONTRAST = 0.55
REGION_POWER = 0.80
IMAGE_MAP_WEIGHT = 0.35
TEMPERATURE_SHIFT = 0.06
GLARE_STRENGTH = 0.08
# =====================================================================


VALID_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp",
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


def calculate_sample_count(
    class_size: int,
    minimum_per_class: int,
    sampling_fraction: float,
) -> int:
    """Use the same weighted rule as the other generators."""
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
    return sorted(path for path in input_root.iterdir() if path.is_dir())


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
        raise FileNotFoundError(f"Input root does not exist: {input_root}")
    if not input_root.is_dir():
        raise NotADirectoryError(
            f"Input root is not a directory: {input_root}"
        )
    if minimum_per_class < 1:
        raise ValueError("minimum_per_class must be at least 1.")
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
            print(f"WARNING: no supported images in {class_directory}")
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
        raise RuntimeError("No images were selected for generation.")

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

    raise ValueError(f"Unsupported image shape: {image.shape}")


def force_odd(value: float, minimum: int = 3) -> int:
    number = max(minimum, int(round(value)))
    return number if number % 2 == 1 else number + 1


def spatial_field(
    height: int,
    width: int,
    pattern: str,
) -> np.ndarray:
    """Create the same broad signed field as the preview program."""
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    if pattern == "diagonal":
        field = 0.5 * (xx + yy)
    elif pattern == "horizontal":
        field = xx
    elif pattern == "vertical":
        field = yy
    elif pattern == "radial":
        radius = np.sqrt(xx**2 + yy**2) / math.sqrt(2.0)
        field = 1.0 - 2.0 * radius
    elif pattern == "vignette":
        radius = np.sqrt(xx**2 + yy**2) / math.sqrt(2.0)
        field = 2.0 * radius - 1.0
    else:
        raise ValueError(
            "PATTERN must be diagonal, horizontal, vertical, "
            "radial, or vignette."
        )

    return np.clip(field, -1.0, 1.0).astype(np.float32)


def image_luminance_field(image: np.ndarray) -> np.ndarray:
    """Estimate broad existing luminance exactly as in the preview."""
    image = ensure_bgr(image)
    height, width = image.shape[:2]
    minimum_dimension = min(height, width)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    luminance = lab[:, :, 0].astype(np.float32) / 255.0

    sigma = float(
        np.clip(
            minimum_dimension * 0.08,
            8.0,
            45.0,
        )
    )
    kernel_size = force_odd(6.0 * sigma + 1.0)

    smooth = cv2.GaussianBlur(
        luminance,
        (kernel_size, kernel_size),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT101,
    )

    low = float(np.percentile(smooth, 5))
    high = float(np.percentile(smooth, 95))

    if high - low < 1e-6:
        return np.zeros_like(smooth, dtype=np.float32)

    normalized = np.clip(
        (smooth - low) / (high - low),
        0.0,
        1.0,
    )

    return (2.0 * normalized - 1.0).astype(np.float32)


def create_illumination_field(image: np.ndarray) -> np.ndarray:
    """Create the same continuous illumination mask as the preview."""
    image = ensure_bgr(image)
    height, width = image.shape[:2]

    geometric = spatial_field(height, width, PATTERN)
    existing = image_luminance_field(image)

    field = (
        (1.0 - IMAGE_MAP_WEIGHT) * geometric
        + IMAGE_MAP_WEIGHT * existing
    ).astype(np.float32)

    sigma = float(
        np.clip(
            min(height, width) * 0.025,
            3.0,
            18.0,
        )
    )
    kernel_size = force_odd(6.0 * sigma + 1.0)

    field = cv2.GaussianBlur(
        field,
        (kernel_size, kernel_size),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT101,
    )

    maximum = float(np.max(np.abs(field)))
    if maximum > 1e-6:
        field = field / maximum

    field = np.clip(field, -1.0, 1.0)

    denominator = math.tanh(FIELD_CONTRAST)
    smooth_field = (
        np.tanh(FIELD_CONTRAST * field) / denominator
    ).astype(np.float32)

    shaped_field = (
        np.sign(smooth_field)
        * np.power(np.abs(smooth_field), REGION_POWER)
    ).astype(np.float32)

    return np.clip(shaped_field, -1.0, 1.0)


def apply_realistic_lighting(
    image: np.ndarray,
    severity: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply the exact approved normalized lighting formula."""
    image = ensure_bgr(image)
    severity = clamp(float(severity), 0.0, 100.0)

    if severity <= 0:
        return image.copy(), {
            "normalized_strength": 0.0,
            "highlight_stops": 0.0,
            "shadow_stops": 0.0,
        }

    normalized = (severity / 100.0) ** SEVERITY_GAMMA
    field = create_illumination_field(image)

    light_shadow_blend = (field + 1.0) / 2.0

    exposure_map = normalized * (
        -SHADOW_STOPS
        + (HIGHLIGHT_STOPS + SHADOW_STOPS)
        * light_shadow_blend
    )

    image_float = image.astype(np.float32) / 255.0

    exposure_multiplier = np.power(
        2.0,
        exposure_map,
    ).astype(np.float32)[:, :, None]

    transformed = image_float * exposure_multiplier

    light_weight = light_shadow_blend
    shadow_weight = 1.0 - light_shadow_blend

    warm = TEMPERATURE_SHIFT * normalized * light_weight
    cool = TEMPERATURE_SHIFT * normalized * shadow_weight

    # OpenCV uses BGR channel order.
    transformed[:, :, 2] *= 1.0 + 0.30 * warm - 0.08 * cool
    transformed[:, :, 1] *= 1.0 + 0.08 * warm
    transformed[:, :, 0] *= 1.0 - 0.15 * warm + 0.20 * cool

    glare_weight = np.clip(field, 0.0, 1.0)
    glare = (
        GLARE_STRENGTH
        * (normalized**2.0)
        * (glare_weight**2.0)
    )[:, :, None]

    transformed = transformed + glare * (1.0 - transformed)

    output_bgr = np.clip(
        transformed * 255.0,
        0.0,
        255.0,
    ).astype(np.uint8)

    return output_bgr, {
        "normalized_strength": normalized,
        "highlight_stops": normalized * HIGHLIGHT_STOPS,
        "shadow_stops": normalized * SHADOW_STOPS,
    }


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

    try:
        encoded.tofile(str(path))
    except OSError as error:
        raise IOError(f"Failed to save: {path}") from error


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a sampled 11-level PlantVillage lighting dataset "
            "using the approved continuous normalized formula."
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
        help="Comma-separated severity levels. Default: 0,10,...,100",
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
        help="Weighted class sampling fraction. Default: 0.086",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed for reproducible sampling. Default: 2026",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG output quality. Default: 95",
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

    return parser.parse_args()


def print_settings() -> None:
    print("\nLOCKED LIGHTING SETTINGS")
    print("=" * 56)
    print(f"PATTERN:             {PATTERN}")
    print(f"SEVERITY_GAMMA:      {SEVERITY_GAMMA}")
    print(f"HIGHLIGHT_STOPS:     {HIGHLIGHT_STOPS}")
    print(f"SHADOW_STOPS:        {SHADOW_STOPS}")
    print(f"FIELD_CONTRAST:      {FIELD_CONTRAST}")
    print(f"REGION_POWER:        {REGION_POWER}")
    print(f"IMAGE_MAP_WEIGHT:    {IMAGE_MAP_WEIGHT}")
    print(f"TEMPERATURE_SHIFT:   {TEMPERATURE_SHIFT}")
    print(f"GLARE_STRENGTH:      {GLARE_STRENGTH}")
    print("=" * 56)


def main() -> None:
    args = parse_args()

    if not 0 <= args.jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be between 0 and 100.")

    _, output_root, severities, selected = prepare_run(
        input_root=args.input_root,
        output_root=args.output_root,
        severities_text=args.severities,
        minimum_per_class=args.minimum_per_class,
        sampling_fraction=args.sampling_fraction,
        seed=args.seed,
    )

    print_settings()

    index_path = output_root / "_lighting_index.csv"
    written = skipped = unreadable = 0

    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "crop",
                "class_name",
                "source_relative_path",
                "output_relative_path",
                "severity",
                "normalized_strength",
                "highlight_stops",
                "shadow_stops",
                "pattern",
                "severity_gamma",
                "field_contrast",
                "region_power",
                "image_map_weight",
                "temperature_shift",
                "glare_strength",
            ]
        )

        for number, row in enumerate(selected, start=1):
            image = cv2.imread(str(row.source_path), cv2.IMREAD_COLOR)

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
                    f"{row.source_path.suffix}"
                )

                output_path = build_output_path(
                    output_root,
                    row.relative_path,
                    output_name,
                )

                if severity > 0:
                    corrupted, parameters = apply_realistic_lighting(
                        image,
                        severity,
                    )
                else:
                    corrupted = None
                    parameters = {
                        "normalized_strength": 0.0,
                        "highlight_stops": 0.0,
                        "shadow_stops": 0.0,
                    }

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
                        output_path.relative_to(output_root).as_posix(),
                        severity,
                        f"{parameters['normalized_strength']:.8f}",
                        f"{parameters['highlight_stops']:.8f}",
                        f"{parameters['shadow_stops']:.8f}",
                        PATTERN,
                        SEVERITY_GAMMA,
                        FIELD_CONTRAST,
                        REGION_POWER,
                        IMAGE_MAP_WEIGHT,
                        TEMPERATURE_SHIFT,
                        GLARE_STRENGTH,
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
                    "source images."
                )

    print("\nLighting generation complete.")
    print(f"Files written: {written:,}")
    print(f"Existing files skipped: {skipped:,}")
    print(f"Unreadable source images: {unreadable:,}")
    print(f"Index: {index_path}")


if __name__ == "__main__":
    main()