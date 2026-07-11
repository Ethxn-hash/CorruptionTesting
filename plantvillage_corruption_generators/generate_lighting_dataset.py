#!/usr/bin/env python3
"""
Generate the sampled PlantVillage severe lighting-variation dataset.

Current repository formulation:
    alpha = S / 100
    B(x,y) = 1 / (1 + exp(-sharpness * (P(x,y) - 0.5)))
    T(x,y) = min_multiplier + (max_multiplier - min_multiplier) * B(x,y)
    M_S(x,y) = (1 - alpha) + alpha * T(x,y)
    I_S(x,y) = clip(I(x,y) * M_S(x,y), 0, 255)
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np

from corruption_common import (
    add_common_arguments,
    build_output_path,
    float_to_filename,
    prepare_run,
    severity_string,
    write_index_header,
)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def lighting_pattern(
    height: int,
    width: int,
    pattern: str = "diagonal",
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
    elif pattern == "radial":
        distance = np.sqrt((xx - 0.5) ** 2 + (yy - 0.5) ** 2)
        maximum_distance = np.sqrt(0.5 ** 2 + 0.5 ** 2)
        values = 1.0 - distance / maximum_distance
    elif pattern == "vignette":
        distance = np.sqrt((xx - 0.5) ** 2 + (yy - 0.5) ** 2)
        maximum_distance = np.sqrt(0.5 ** 2 + 0.5 ** 2)
        values = distance / maximum_distance
    else:
        raise ValueError(
            "pattern must be diagonal, horizontal, vertical, radial, "
            "or vignette"
        )

    return np.clip(values, 0, 1).astype(np.float32)


def apply_lighting_variation(
    image: np.ndarray,
    severity: float,
    min_multiplier: float = 0.00,
    max_multiplier: float = 6.00,
    sharpness: float = 12.0,
    pattern: str = "diagonal",
) -> np.ndarray:
    height, width = image.shape[:2]
    alpha = clamp(severity, 0, 100) / 100.0
    spatial_pattern = lighting_pattern(height, width, pattern)

    harsh_split = 1.0 / (
        1.0 + np.exp(-sharpness * (spatial_pattern - 0.5))
    )
    target_multiplier = (
        min_multiplier
        + (max_multiplier - min_multiplier) * harsh_split
    )
    multiplier = (1.0 - alpha) + alpha * target_multiplier

    if image.ndim == 3:
        multiplier = multiplier[:, :, None]

    corrupted = image.astype(np.float32) * multiplier
    return np.clip(corrupted, 0, 255).astype(np.uint8)


def save_cv_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise IOError(f"OpenCV failed to save: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a sampled 11-level PlantVillage lighting-variation "
            "dataset while preserving the clean folder structure."
        )
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--min-multiplier",
        type=float,
        default=0.00,
        help="Darkest multiplier at severity 100. Default: 0.00.",
    )
    parser.add_argument(
        "--max-multiplier",
        type=float,
        default=6.00,
        help="Brightest multiplier at severity 100. Default: 6.00.",
    )
    parser.add_argument(
        "--sharpness",
        type=float,
        default=12.0,
        help="Sharpness of the dark/bright transition. Default: 12.0.",
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
        help="Spatial lighting pattern. Default: diagonal.",
    )
    args = parser.parse_args()

    if args.max_multiplier < args.min_multiplier:
        parser.error(
            "--max-multiplier must be greater than or equal to "
            "--min-multiplier."
        )
    if args.sharpness <= 0:
        parser.error("--sharpness must be positive.")

    _, output_root, severities, selected = prepare_run(
        args,
        factor_label="lighting",
    )

    index_path = output_root / "_lighting_index.csv"
    generated = 0
    skipped_existing = 0
    unreadable = 0

    min_tag = float_to_filename(args.min_multiplier, 2)
    max_tag = float_to_filename(args.max_multiplier, 2)
    sharpness_tag = float_to_filename(args.sharpness, 2)

    with index_path.open("w", newline="", encoding="utf-8") as index_handle:
        writer = csv.writer(index_handle)
        write_index_header(
            writer,
            [
                "min_multiplier",
                "max_multiplier",
                "sharpness",
                "pattern",
            ],
        )

        for source_number, row in enumerate(selected, start=1):
            source_path = row.source_path
            image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)

            if image is None:
                print(f"WARNING: unreadable image skipped: {source_path}")
                unreadable += 1
                continue

            for severity in severities:
                output_name = (
                    f"{source_path.stem}"
                    f"_lighting_s{severity_string(severity)}"
                    f"_min{min_tag}"
                    f"_max{max_tag}"
                    f"_sh{sharpness_tag}"
                    f"_{args.pattern}"
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
                    shutil.copy2(source_path, output_path)
                    generated += 1
                else:
                    corrupted = apply_lighting_variation(
                        image=image,
                        severity=severity,
                        min_multiplier=args.min_multiplier,
                        max_multiplier=args.max_multiplier,
                        sharpness=args.sharpness,
                        pattern=args.pattern,
                    )
                    save_cv_image(output_path, corrupted)
                    generated += 1

                writer.writerow(
                    [
                        row.class_name,
                        row.relative_path.as_posix(),
                        output_path.relative_to(output_root).as_posix(),
                        severity,
                        args.min_multiplier,
                        args.max_multiplier,
                        args.sharpness,
                        args.pattern,
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

    print("\nLighting generation complete.")
    print(f"New files written: {generated:,}")
    print(f"Existing files skipped: {skipped_existing:,}")
    print(f"Unreadable source images: {unreadable:,}")
    print(f"Output index: {index_path}")


if __name__ == "__main__":
    main()
