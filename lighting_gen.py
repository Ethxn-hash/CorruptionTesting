#!/usr/bin/env python3
"""
Generate the full harsh-illumination corruption experiment.

Expected layout:
    input_root/
        Class_A/
            image1.jpg
        Class_B/
            image2.jpg

Output layout:
    output_root/
        severity_000/Class_A/image1_lighting_s000.png
        severity_010/Class_A/image1_lighting_s010.png
        ...
        severity_100/Class_A/image1_lighting_s100.png
        lighting_metadata.csv

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
            "Generate harsh-illumination corruptions at severities "
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


def ensure_bgr(image):
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


def force_odd(value):
    value = max(3, int(round(value)))

    if value % 2 == 0:
        value += 1

    return value


def create_lighting_map(image):
    """
    Build a broad illumination map from the image's original luminance.

    This makes already-bright regions brighter and already-dark regions
    darker without introducing a synthetic geometric split.
    """

    image = ensure_bgr(image)
    height, width = image.shape[:2]
    minimum_dimension = min(height, width)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    luminance = lab[:, :, 0].astype(np.float32) / 255.0

    sigma = float(
        np.clip(
            minimum_dimension * 0.035,
            3.0,
            35.0,
        )
    )

    kernel_size = force_odd(6.0 * sigma + 1.0)

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
            smoothed - low
        ) / (
            high - low
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


def apply_lighting_degradation(image, severity):
    """
    Severity 0:
        Pixel-identical decoded image.

    Increasing severity:
        Broad bright regions move toward white.
        Broad dark regions move toward black.
        The output gradually blends toward a binary exposure endpoint.

    Severity 100:
        Pure black-and-white, three-channel image.
    """

    image = ensure_bgr(image)

    if severity == 0:
        return image.copy(), 0.0, 0.0

    push = float(ns.get_lighting_push(severity))
    bw_blend = float(ns.get_lighting_bw_blend(severity))

    push = float(np.clip(push, 0.0, 1.5))
    bw_blend = float(np.clip(bw_blend, 0.0, 1.0))

    image_float = image.astype(np.float32)

    normalized_map, signed_map = create_lighting_map(image)

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

    illumination_image = (
        image_float
        + bright_change
        - dark_change
    )

    illumination_image = np.clip(
        illumination_image,
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

    if severity == 100:
        corrupted = binary_target.copy()

    corrupted = np.clip(
        corrupted,
        0.0,
        255.0,
    ).astype(np.uint8)

    return ensure_bgr(corrupted), push, bw_blend


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
        f"{input_path.stem}_lighting_s{severity:03d}.png"
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

    metadata_path = output_root / "lighting_metadata.csv"

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
                "lighting_push",
                "black_white_blend",
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
                            "factor": "lighting",
                            "severity": severity,
                            "lighting_push": "",
                            "black_white_blend": "",
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
                push = ""
                bw_blend = ""

                try:
                    if severity == 0:
                        push = 0.0
                        bw_blend = 0.0
                    else:
                        push = float(
                            ns.get_lighting_push(severity)
                        )
                        bw_blend = float(
                            ns.get_lighting_bw_blend(severity)
                        )

                    if output_path.exists() and not args.overwrite:
                        status = "skipped_existing"
                        skipped += 1
                    else:
                        (
                            corrupted,
                            push,
                            bw_blend,
                        ) = apply_lighting_degradation(
                            image,
                            severity,
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
                        "factor": "lighting",
                        "severity": severity,
                        "lighting_push": push,
                        "black_white_blend": bw_blend,
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

    print("\nLighting generation complete.")
    print(f"Source images: {len(images)}")
    print(f"Expected outputs: {total_expected}")
    print(f"Written: {written}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()