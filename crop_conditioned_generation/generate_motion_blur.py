#!/usr/bin/env python3
"""Generate motion-blurred images for only Apple, Corn, Grape, and Tomato."""

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


def nearest_odd(value: float) -> int:
    number = max(1, int(round(value)))
    return number if number % 2 == 1 else number + 1


def kernel_size(severity: float, height: int, width: int) -> int:
    maximum = nearest_odd(
        float(config.MOTION_BLUR_MAX_FRACTION) * min(height, width)
    )
    alpha = (clamp(severity, 0, 100) / 100.0) ** float(
        config.MOTION_BLUR_GAMMA
    )
    return nearest_odd(1 + (maximum - 1) * alpha)


def motion_kernel(size: int, angle_degrees: float) -> np.ndarray:
    if size <= 1:
        return np.array([[1.0]], dtype=np.float32)
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    kernel[center, :] = 1.0
    matrix = cv2.getRotationMatrix2D((center, center), angle_degrees, 1.0)
    rotated = cv2.warpAffine(kernel, matrix, (size, size))
    total = float(rotated.sum())
    if total <= 0:
        raise RuntimeError("Motion-blur kernel has zero total weight.")
    return rotated / total


def main() -> None:
    _, output_root, severities, selected = prepare_generation(
        config, "motion_blur"
    )
    index_path = output_root / "_motion_blur_index.csv"
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
                "kernel_size",
                "angle_degrees",
                "max_blur_fraction",
                "gamma",
            ]
        )

        for number, row in enumerate(selected, 1):
            image = cv2.imread(str(row.source_path), cv2.IMREAD_COLOR)
            if image is None:
                print(f"WARNING: unreadable image skipped: {row.source_path}")
                unreadable += 1
                continue
            height, width = image.shape[:2]

            for severity in severities:
                size = kernel_size(severity, height, width)
                output_name = (
                    f"{row.source_path.stem}_motion_s{severity_string(severity)}"
                    f"_k{size}_a{float_tag(config.MOTION_BLUR_ANGLE_DEGREES, 1)}"
                    f"{row.source_path.suffix}"
                )
                output_path = build_output_path(
                    output_root, row.relative_path, output_name
                )
                corrupted = None
                if severity > 0:
                    kernel = motion_kernel(
                        size, float(config.MOTION_BLUR_ANGLE_DEGREES)
                    )
                    corrupted = cv2.filter2D(image, -1, kernel)
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
                        size,
                        config.MOTION_BLUR_ANGLE_DEGREES,
                        config.MOTION_BLUR_MAX_FRACTION,
                        config.MOTION_BLUR_GAMMA,
                    ]
                )

            if config.PROGRESS_EVERY > 0 and (
                number % config.PROGRESS_EVERY == 0 or number == len(selected)
            ):
                print(f"Processed {number:,}/{len(selected):,} source images.")

    print("\nMotion-blur generation complete.")
    print(f"Files written: {written:,}")
    print(f"Existing files skipped: {skipped:,}")
    print(f"Unreadable source images: {unreadable:,}")
    print(f"Index: {index_path}")


if __name__ == "__main__":
    main()
