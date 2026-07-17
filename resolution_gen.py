#!/usr/bin/env python3
"""
Generate the full resolution-degradation experiment.

Expected layout:
    input_root/
        Class_A/
            image1.jpg
        Class_B/
            image2.jpg

Output layout:
    output_root/
        severity_000/Class_A/image1_resolution_s000.png
        severity_010/Class_A/image1_resolution_s010.png
        ...
        severity_100/Class_A/image1_resolution_s100.png
        resolution_metadata.csv

Place normalized_severity.py in the same directory as this script.
"""

import argparse
import csv
from pathlib import Path

import cv2

import normalized_severity as ns


SEVERITIES = list(range(0, 101, 10))
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate resolution corruptions at severities "
            "0, 10, 20, ..., 100."
        )
    )
    parser.add_argument(
        "input_root",
        type=Path,
        help="Dataset root containing class folders.",
    )
    parser.add_argument(
        "output_root",
        type=Path,
        help="Directory where corrupted datasets will be written.",
    )
    parser.add_argument(
        "--upsample_method",
        choices=["nearest", "linear", "cubic"],
        default="linear",
        help=(
            "Interpolation used to restore the original dimensions. "
            "Default: linear."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate outputs that already exist.",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=100,
        help="Print progress after this many source images. Default: 100.",
    )
    return parser.parse_args()


def find_images(input_root, output_root):
    input_root = input_root.resolve()
    output_root = output_root.resolve()

    images = []

    for path in input_root.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        try:
            path.resolve().relative_to(output_root)
            continue
        except ValueError:
            pass

        images.append(path)

    return sorted(images, key=lambda p: str(p.relative_to(input_root)).lower())


def read_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"OpenCV could not read image: {path}")

    return image


def apply_resolution_degradation(
    image,
    severity,
    upsample_method,
):
    """
    Severity 0:
        Pixel-identical decoded image.

    Severities 10-90:
        Downsample using the normalized scale, then restore the original
        dimensions.

    Severity 100:
        Downsample to exactly 1 x 1 pixel, then restore.
    """

    height, width = image.shape[:2]

    if severity == 0:
        return image.copy(), 1.0, width, height

    if severity == 100:
        scale = 0.0
        reduced_width = 1
        reduced_height = 1
    else:
        scale = float(ns.get_resolution_scale(severity))
        scale = max(0.0, min(1.0, scale))

        reduced_width = max(
            1,
            int(round(width * scale)),
        )

        reduced_height = max(
            1,
            int(round(height * scale)),
        )

    reduced = cv2.resize(
        image,
        (reduced_width, reduced_height),
        interpolation=cv2.INTER_AREA,
    )

    interpolation_map = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
    }

    restored = cv2.resize(
        reduced,
        (width, height),
        interpolation=interpolation_map[upsample_method],
    )

    return (
        restored,
        scale,
        reduced_width,
        reduced_height,
    )


def make_output_path(
    input_path,
    input_root,
    output_root,
    severity,
):
    relative = input_path.relative_to(input_root)
    relative_parent = relative.parent

    output_directory = (
        output_root
        / f"severity_{severity:03d}"
        / relative_parent
    )

    output_directory.mkdir(parents=True, exist_ok=True)

    output_name = (
        f"{input_path.stem}_resolution_s{severity:03d}.png"
    )

    return output_directory / output_name


def main():
    args = parse_args()

    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not input_root.is_dir():
        raise NotADirectoryError(
            f"Input root does not exist or is not a directory: {input_root}"
        )

    output_root.mkdir(parents=True, exist_ok=True)

    images = find_images(input_root, output_root)

    if not images:
        raise FileNotFoundError(
            f"No supported images found under: {input_root}"
        )

    metadata_path = output_root / "resolution_metadata.csv"

    total_expected = len(images) * len(SEVERITIES)
    written = 0
    skipped = 0
    failed = 0

    with metadata_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as metadata_file:
        writer = csv.DictWriter(
            metadata_file,
            fieldnames=[
                "source_relative_path",
                "output_relative_path",
                "class_path",
                "factor",
                "severity",
                "scale_factor",
                "reduced_width",
                "reduced_height",
                "upsample_method",
                "status",
                "error",
            ],
        )
        writer.writeheader()

        for image_index, input_path in enumerate(images, start=1):
            relative = input_path.relative_to(input_root)
            class_path = str(relative.parent)

            try:
                image = read_image(input_path)
            except Exception as error:
                failed += len(SEVERITIES)

                for severity in SEVERITIES:
                    writer.writerow(
                        {
                            "source_relative_path": str(relative),
                            "output_relative_path": "",
                            "class_path": class_path,
                            "factor": "resolution",
                            "severity": severity,
                            "scale_factor": "",
                            "reduced_width": "",
                            "reduced_height": "",
                            "upsample_method": args.upsample_method,
                            "status": "failed",
                            "error": str(error),
                        }
                    )

                continue

            height, width = image.shape[:2]

            for severity in SEVERITIES:
                output_path = make_output_path(
                    input_path,
                    input_root,
                    output_root,
                    severity,
                )

                status = "written"
                error_text = ""
                scale = ""
                reduced_width = ""
                reduced_height = ""

                try:
                    if output_path.exists() and not args.overwrite:
                        status = "skipped_existing"
                        skipped += 1

                        if severity == 0:
                            scale = 1.0
                            reduced_width = width
                            reduced_height = height
                        elif severity == 100:
                            scale = 0.0
                            reduced_width = 1
                            reduced_height = 1
                        else:
                            scale = float(
                                ns.get_resolution_scale(severity)
                            )
                            scale = max(0.0, min(1.0, scale))
                            reduced_width = max(
                                1,
                                int(round(width * scale)),
                            )
                            reduced_height = max(
                                1,
                                int(round(height * scale)),
                            )
                    else:
                        (
                            corrupted,
                            scale,
                            reduced_width,
                            reduced_height,
                        ) = apply_resolution_degradation(
                            image,
                            severity,
                            args.upsample_method,
                        )

                        success = cv2.imwrite(
                            str(output_path),
                            corrupted,
                            [cv2.IMWRITE_PNG_COMPRESSION, 3],
                        )

                        if not success:
                            raise IOError(
                                f"cv2.imwrite failed for {output_path}"
                            )

                        written += 1

                except Exception as error:
                    status = "failed"
                    error_text = str(error)
                    failed += 1

                writer.writerow(
                    {
                        "source_relative_path": str(relative),
                        "output_relative_path": (
                            str(output_path.relative_to(output_root))
                            if status != "failed"
                            else ""
                        ),
                        "class_path": class_path,
                        "factor": "resolution",
                        "severity": severity,
                        "scale_factor": scale,
                        "reduced_width": reduced_width,
                        "reduced_height": reduced_height,
                        "upsample_method": args.upsample_method,
                        "status": status,
                        "error": error_text,
                    }
                )

            if (
                args.progress_every > 0
                and (
                    image_index % args.progress_every == 0
                    or image_index == len(images)
                )
            ):
                print(
                    f"[{image_index}/{len(images)} source images] "
                    f"written={written}, skipped={skipped}, failed={failed}"
                )

    print("\nResolution generation complete.")
    print(f"Source images: {len(images)}")
    print(f"Expected outputs: {total_expected}")
    print(f"Written: {written}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()