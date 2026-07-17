#!/usr/bin/env python3
"""
Generate the full motion-blur corruption experiment.

Expected layout:
    input_root/
        Class_A/
            image1.jpg
        Class_B/
            image2.jpg

Output layout:
    output_root/
        severity_000/Class_A/image1_blur_s000.png
        severity_010/Class_A/image1_blur_s010.png
        ...
        severity_100/Class_A/image1_blur_s100.png
        blur_metadata.csv

Place normalized_severity.py in the same directory as this script.
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

import normalized_severity as ns


SEVERITIES = list(range(0, 101, 10))
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate motion-blur corruptions at severities "
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
        "--angle",
        type=float,
        default=15.0,
        help=(
            "Fixed motion-blur angle in degrees. "
            "Default: 15.0. Keep fixed to isolate blur magnitude."
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

        # Prevent accidental reprocessing if output_root is inside input_root.
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


def force_odd(value):
    value = max(1, int(round(value)))

    if value % 2 == 0:
        value += 1

    return value


def make_motion_blur_kernel(length, angle_degrees):
    length = force_odd(length)

    if length <= 1:
        return np.array([[1.0]], dtype=np.float32)

    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0

    center = ((length - 1) / 2.0, (length - 1) / 2.0)

    rotation = cv2.getRotationMatrix2D(
        center,
        angle_degrees,
        1.0,
    )

    kernel = cv2.warpAffine(
        kernel,
        rotation,
        (length, length),
        flags=cv2.INTER_LINEAR,
    )

    kernel_sum = float(kernel.sum())

    if kernel_sum <= 0:
        return np.array([[1.0]], dtype=np.float32)

    return kernel / kernel_sum


def apply_motion_blur(image, severity, angle_degrees):
    """
    Severity 0:
        Pixel-identical decoded image.

    Severities 10-90:
        Motion blur with kernel length from normalized_severity.py.

    Severity 100:
        Complete spatial-information collapse to the image mean color.
    """

    if severity == 0:
        return image.copy(), 1

    if severity == 100:
        mean_color = np.mean(
            image.astype(np.float32),
            axis=(0, 1),
            keepdims=True,
        )

        collapsed = np.broadcast_to(
            mean_color,
            image.shape,
        ).copy()

        return np.clip(collapsed, 0, 255).astype(np.uint8), 0

    requested_length = ns.get_blur_length(severity)
    requested_length = force_odd(requested_length)

    maximum_length = min(image.shape[:2])

    if maximum_length % 2 == 0:
        maximum_length -= 1

    maximum_length = max(1, maximum_length)
    kernel_length = min(requested_length, maximum_length)

    kernel = make_motion_blur_kernel(
        kernel_length,
        angle_degrees,
    )

    corrupted = cv2.filter2D(
        image,
        ddepth=-1,
        kernel=kernel,
        borderType=cv2.BORDER_REFLECT101,
    )

    return corrupted, kernel_length


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
        f"{input_path.stem}_blur_s{severity:03d}.png"
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

    metadata_path = output_root / "blur_metadata.csv"

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
                "kernel_length",
                "angle_degrees",
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
                            "factor": "motion_blur",
                            "severity": severity,
                            "kernel_length": "",
                            "angle_degrees": args.angle,
                            "status": "failed",
                            "error": str(error),
                        }
                    )

                continue

            for severity in SEVERITIES:
                output_path = make_output_path(
                    input_path,
                    input_root,
                    output_root,
                    severity,
                )

                status = "written"
                error_text = ""
                kernel_length = ""

                try:
                    if output_path.exists() and not args.overwrite:
                        status = "skipped_existing"
                        skipped += 1

                        if severity == 100:
                            kernel_length = 0
                        elif severity == 0:
                            kernel_length = 1
                        else:
                            maximum_length = min(image.shape[:2])
                            if maximum_length % 2 == 0:
                                maximum_length -= 1
                            kernel_length = min(
                                force_odd(ns.get_blur_length(severity)),
                                max(1, maximum_length),
                            )
                    else:
                        corrupted, kernel_length = apply_motion_blur(
                            image,
                            severity,
                            args.angle,
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
                        "factor": "motion_blur",
                        "severity": severity,
                        "kernel_length": kernel_length,
                        "angle_degrees": args.angle,
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

    print("\nMotion-blur generation complete.")
    print(f"Source images: {len(images)}")
    print(f"Expected outputs: {total_expected}")
    print(f"Written: {written}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()