#!/usr/bin/env python3
"""Generate resolution-degraded images for the configured crop classes."""

from __future__ import annotations

import csv

import cv2

import generation_config as config
from crop_corruption_common import (
    build_output_path,
    clamp,
    float_tag,
    prepare_generation,
    severity_string,
    write_or_copy,
)


def scale_factor(severity: float) -> float:
    minimum = float(config.RESOLUTION_MIN_SCALE)
    if not 0 < minimum <= 1:
        raise ValueError("RESOLUTION_MIN_SCALE must be in (0, 1].")
    scale = 1.0 - (1.0 - minimum) * (clamp(severity, 0, 100) / 100.0)
    return clamp(scale, minimum, 1.0)


def degrade(image, severity: float):
    height, width = image.shape[:2]
    scale = scale_factor(severity)
    small_width = max(1, int(round(width * scale)))
    small_height = max(1, int(round(height * scale)))
    reduced = cv2.resize(
        image, (small_width, small_height), interpolation=cv2.INTER_AREA
    )
    methods = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
    }
    method_name = str(config.RESOLUTION_UPSAMPLE_METHOD).lower()
    if method_name not in methods:
        raise ValueError(
            "RESOLUTION_UPSAMPLE_METHOD must be nearest, linear, or cubic."
        )
    restored = cv2.resize(
        reduced, (width, height), interpolation=methods[method_name]
    )
    return restored, scale, small_width, small_height


def main() -> None:
    _, output_root, severities, selected = prepare_generation(
        config, "resolution"
    )
    index_path = output_root / "_resolution_index.csv"
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
                "scale_factor",
                "downsample_width",
                "downsample_height",
                "upsample_method",
                "minimum_scale",
            ]
        )

        for number, row in enumerate(selected, 1):
            image = cv2.imread(str(row.source_path), cv2.IMREAD_COLOR)
            if image is None:
                print(f"WARNING: unreadable image skipped: {row.source_path}")
                unreadable += 1
                continue

            for severity in severities:
                height, width = image.shape[:2]
                scale = scale_factor(severity)
                small_width = max(1, int(round(width * scale)))
                small_height = max(1, int(round(height * scale)))
                output_name = (
                    f"{row.source_path.stem}_resolution_s{severity_string(severity)}"
                    f"_Sc{float_tag(scale, 4)}{row.source_path.suffix}"
                )
                output_path = build_output_path(
                    output_root, row.relative_path, output_name
                )
                corrupted = None
                if severity > 0:
                    corrupted, scale, small_width, small_height = degrade(
                        image, severity
                    )
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
                        f"{scale:.8f}",
                        small_width,
                        small_height,
                        config.RESOLUTION_UPSAMPLE_METHOD,
                        config.RESOLUTION_MIN_SCALE,
                    ]
                )

            if config.PROGRESS_EVERY > 0 and (
                number % config.PROGRESS_EVERY == 0 or number == len(selected)
            ):
                print(f"Processed {number:,}/{len(selected):,} source images.")

    print("\nResolution generation complete.")
    print(f"Files written: {written:,}")
    print(f"Existing files skipped: {skipped:,}")
    print(f"Unreadable source images: {unreadable:,}")
    print(f"Index: {index_path}")


if __name__ == "__main__":
    main()
