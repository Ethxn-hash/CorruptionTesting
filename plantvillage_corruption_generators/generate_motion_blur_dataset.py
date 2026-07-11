#!/usr/bin/env python3
"""
Generate the sampled PlantVillage motion-blur dataset.

Formula used by the current repository implementation:
    L_max = nearest_odd(max_blur_fraction * min(H, W))
    k(S)  = nearest_odd(1 + (L_max - 1) * (S / 100)^gamma)

Default severities: 0, 10, ..., 100.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from corruption_common import (
    add_common_arguments,
    build_output_path,
    prepare_run,
    severity_string,
    write_index_header,
)


def nearest_odd(value: float) -> int:
    number = max(1, int(round(value)))
    if number % 2 == 0:
        number += 1
    return number


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


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


def apply_motion_blur(
    image: np.ndarray,
    severity: float,
    angle_degrees: float = 0.0,
    max_blur_fraction: float = 0.30,
    gamma: float = 2.0,
) -> Tuple[np.ndarray, int]:
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
            "Severity-100 blur length as a fraction of min(H, W). "
            "Default: 0.30."
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
