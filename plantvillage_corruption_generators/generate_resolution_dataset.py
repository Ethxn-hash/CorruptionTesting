#!/usr/bin/env python3
"""
Generate the sampled PlantVillage resolution-degradation dataset.

Formula used by the current repository implementation:
    scale(S) = 1 - (1 - min_scale) * (S / 100)

The image is downsampled with INTER_AREA and resized back to its original
dimensions using the selected upsampling method.
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
    float_to_filename,
    prepare_run,
    severity_string,
    write_index_header,
)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def resolution_scale_factor(
    severity: float,
    min_scale: float = 0.03,
) -> float:
    severity = clamp(severity, 0, 100)
    scale = 1.0 - (1.0 - min_scale) * (severity / 100.0)
    return clamp(scale, min_scale, 1.0)


def apply_resolution_degradation(
    image: np.ndarray,
    severity: float,
    min_scale: float = 0.03,
    upsample_method: str = "linear",
) -> Tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    scale = resolution_scale_factor(severity, min_scale=min_scale)

    small_width = max(1, int(round(width * scale)))
    small_height = max(1, int(round(height * scale)))

    reduced = cv2.resize(
        image,
        (small_width, small_height),
        interpolation=cv2.INTER_AREA,
    )

    interpolation_methods = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
    }
    try:
        interpolation = interpolation_methods[upsample_method]
    except KeyError as exc:
        raise ValueError(
            "upsample_method must be nearest, linear, or cubic"
        ) from exc

    degraded = cv2.resize(
        reduced,
        (width, height),
        interpolation=interpolation,
    )
    return degraded, scale


def save_cv_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise IOError(f"OpenCV failed to save: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a sampled 11-level PlantVillage resolution-degradation "
            "dataset while preserving the clean folder structure."
        )
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--min-scale",
        type=float,
        default=0.03,
        help="Downsampling scale at severity 100. Default: 0.03.",
    )
    parser.add_argument(
        "--upsample-method",
        choices=["nearest", "linear", "cubic"],
        default="linear",
        help="Interpolation used to return to the original dimensions.",
    )
    args = parser.parse_args()

    if not 0 < args.min_scale <= 1:
        parser.error("--min-scale must be in the interval (0, 1].")

    _, output_root, severities, selected = prepare_run(
        args,
        factor_label="resolution",
    )

    index_path = output_root / "_resolution_index.csv"
    generated = 0
    skipped_existing = 0
    unreadable = 0

    with index_path.open("w", newline="", encoding="utf-8") as index_handle:
        writer = csv.writer(index_handle)
        write_index_header(
            writer,
            ["scale_factor", "upsample_method", "min_scale"],
        )

        for source_number, row in enumerate(selected, start=1):
            source_path = row.source_path
            image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)

            if image is None:
                print(f"WARNING: unreadable image skipped: {source_path}")
                unreadable += 1
                continue

            for severity in severities:
                scale = resolution_scale_factor(
                    severity,
                    min_scale=args.min_scale,
                )
                output_name = (
                    f"{source_path.stem}"
                    f"_resolution_s{severity_string(severity)}"
                    f"_Sc{float_to_filename(scale, 4)}"
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
                    corrupted, _ = apply_resolution_degradation(
                        image=image,
                        severity=severity,
                        min_scale=args.min_scale,
                        upsample_method=args.upsample_method,
                    )
                    save_cv_image(output_path, corrupted)
                    generated += 1

                writer.writerow(
                    [
                        row.class_name,
                        row.relative_path.as_posix(),
                        output_path.relative_to(output_root).as_posix(),
                        severity,
                        f"{scale:.8f}",
                        args.upsample_method,
                        args.min_scale,
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

    print("\nResolution generation complete.")
    print(f"New files written: {generated:,}")
    print(f"Existing files skipped: {skipped_existing:,}")
    print(f"Unreadable source images: {unreadable:,}")
    print(f"Output index: {index_path}")


if __name__ == "__main__":
    main()
