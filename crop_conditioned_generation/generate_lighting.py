#!/usr/bin/env python3
"""Generate lighting-corrupted images for the configured crop classes."""

from __future__ import annotations

import csv

import cv2
import numpy as np

import generation_config as config
from crop_corruption_common import (
    build_output_path,
    clamp,
    float_tag,
    prepare_generation,
    severity_string,
    write_or_copy,
)


def spatial_pattern(height: int, width: int, pattern: str) -> np.ndarray:
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
        distance = np.sqrt((xx - 0.5) ** 2 + (yy - 0.5) ** 2)
        maximum = np.sqrt(0.5**2 + 0.5**2)
        values = 1.0 - distance / maximum
        if pattern == "vignette":
            values = 1.0 - values
    else:
        raise ValueError(
            "LIGHTING_PATTERN must be diagonal, horizontal, vertical, "
            "radial, or vignette."
        )
    return np.clip(values, 0, 1).astype(np.float32)


def apply_lighting(image: np.ndarray, severity: float) -> np.ndarray:
    minimum = float(config.LIGHTING_MIN_MULTIPLIER)
    maximum = float(config.LIGHTING_MAX_MULTIPLIER)
    sharpness = float(config.LIGHTING_SHARPNESS)
    if maximum < minimum:
        raise ValueError(
            "LIGHTING_MAX_MULTIPLIER must be >= LIGHTING_MIN_MULTIPLIER."
        )
    if sharpness <= 0:
        raise ValueError("LIGHTING_SHARPNESS must be positive.")

    height, width = image.shape[:2]
    alpha = clamp(severity, 0, 100) / 100.0
    pattern = spatial_pattern(height, width, str(config.LIGHTING_PATTERN))
    split = 1.0 / (1.0 + np.exp(-sharpness * (pattern - 0.5)))
    target = minimum + (maximum - minimum) * split
    multiplier = (1.0 - alpha) + alpha * target
    if image.ndim == 3:
        multiplier = multiplier[:, :, None]
    result = image.astype(np.float32) * multiplier
    return np.clip(result, 0, 255).astype(np.uint8)


def main() -> None:
    _, output_root, severities, selected = prepare_generation(config, "lighting")
    index_path = output_root / "_lighting_index.csv"
    written = skipped = unreadable = 0

    min_tag = float_tag(config.LIGHTING_MIN_MULTIPLIER, 2)
    max_tag = float_tag(config.LIGHTING_MAX_MULTIPLIER, 2)
    sharp_tag = float_tag(config.LIGHTING_SHARPNESS, 2)

    with index_path.open("w", newline="", encoding="utf-8") as handle:
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

        for number, row in enumerate(selected, 1):
            image = cv2.imread(str(row.source_path), cv2.IMREAD_COLOR)
            if image is None:
                print(f"WARNING: unreadable image skipped: {row.source_path}")
                unreadable += 1
                continue

            for severity in severities:
                output_name = (
                    f"{row.source_path.stem}_lighting_s{severity_string(severity)}"
                    f"_min{min_tag}_max{max_tag}_sh{sharp_tag}"
                    f"_{config.LIGHTING_PATTERN}{row.source_path.suffix}"
                )
                output_path = build_output_path(
                    output_root, row.relative_path, output_name
                )
                corrupted = apply_lighting(image, severity) if severity > 0 else None
                status = write_or_copy(
                    row.source_path,
                    output_path,
                    severity,
                    corrupted,
                    bool(config.OVERWRITE),
                    int(config.JPEG_QUALITY),
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
                        config.LIGHTING_MIN_MULTIPLIER,
                        config.LIGHTING_MAX_MULTIPLIER,
                        config.LIGHTING_SHARPNESS,
                        config.LIGHTING_PATTERN,
                    ]
                )

            if config.PROGRESS_EVERY > 0 and (
                number % config.PROGRESS_EVERY == 0 or number == len(selected)
            ):
                print(f"Processed {number:,}/{len(selected):,} source images.")

    print("\nLighting generation complete.")
    print(f"Files written: {written:,}")
    print(f"Existing files skipped: {skipped:,}")
    print(f"Unreadable source images: {unreadable:,}")
    print(f"Index: {index_path}")


if __name__ == "__main__":
    main()
