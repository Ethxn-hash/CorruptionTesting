#!/usr/bin/env python3
"""
Normalized resolution-degradation generator.

Key behavior
------------
1. Preserves the original true-label class folders.
2. Uses weighted class sampling with a minimum of 100 images per class:
       n_c = min(N_c, max(minimum_per_class,
                          round(sampling_fraction * N_c)))
3. Uses 11 normalized severity levels by default:
       0, 10, 20, ..., 100
4. Generates every severity level for every selected source image.
5. Uses the same seed and sampling algorithm as the other generators.
6. Writes both a generation metadata CSV and a selected-sources CSV.
"""

import argparse
import csv
import hashlib
import random
from pathlib import Path

import cv2
import numpy as np


VALID_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp"
}

INTERPOLATION_METHODS = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate normalized resolution-degradation images using "
            "weighted class sampling while preserving true-label folders."
        )
    )

    parser.add_argument(
        "--input_root",
        type=Path,
        required=True,
        help="Dataset root containing one folder per true-label class."
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        required=True,
        help="Output root for generated resolution-degradation images."
    )
    parser.add_argument(
        "--minimum_per_class",
        type=int,
        default=100,
        help="Minimum number of source images selected per class when available."
    )
    parser.add_argument(
        "--sampling_fraction",
        type=float,
        default=0.086,
        help="Proportional sampling fraction for larger classes."
    )
    parser.add_argument(
        "--severity_levels",
        type=str,
        default="0,10,20,30,40,50,60,70,80,90,100",
        help="Comma-separated severity levels from 0 to 100."
    )
    parser.add_argument(
        "--min_scale",
        type=float,
        default=0.03,
        help="Retained width/height scale at severity 100."
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help=(
            "Exponent applied to normalized severity. The default 1.0 "
            "keeps the original linear normalized-severity mapping."
        )
    )
    parser.add_argument(
        "--upsample_method",
        choices=sorted(INTERPOLATION_METHODS),
        default="bilinear",
        help="Interpolation used to resize the low-resolution image back."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed for reproducible weighted class sampling."
    )
    parser.add_argument(
        "--metadata_csv",
        type=str,
        default="resolution_metadata.csv",
        help="Generation metadata CSV filename."
    )
    parser.add_argument(
        "--selection_csv",
        type=str,
        default="resolution_selected_sources.csv",
        help="CSV recording selected original images."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an output image when it already exists."
    )

    return parser.parse_args()


def stable_integer(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def parse_severity_levels(text: str):
    levels = sorted({
        int(value.strip())
        for value in text.split(",")
        if value.strip()
    })

    if not levels:
        raise ValueError("At least one severity level must be provided.")

    for severity in levels:
        if not 0 <= severity <= 100:
            raise ValueError(
                f"Severity {severity} is outside the valid range 0-100."
            )

    return levels


def normalized_severity(severity: int) -> float:
    return float(np.clip(severity, 0, 100)) / 100.0


def calculate_sample_count(
    class_size: int,
    minimum_per_class: int,
    sampling_fraction: float
) -> int:
    if class_size <= 0:
        return 0

    proportional_count = round(class_size * sampling_fraction)
    requested_count = max(minimum_per_class, proportional_count)
    return min(class_size, requested_count)


def find_class_directories(input_root: Path):
    return sorted(
        path for path in input_root.iterdir()
        if path.is_dir()
    )


def find_images(class_directory: Path):
    return sorted(
        path for path in class_directory.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    )


def select_weighted_images(
    images,
    class_name: str,
    minimum_per_class: int,
    sampling_fraction: float,
    seed: int
):
    sample_count = calculate_sample_count(
        class_size=len(images),
        minimum_per_class=minimum_per_class,
        sampling_fraction=sampling_fraction
    )

    if sample_count >= len(images):
        return list(images)

    rng = random.Random(seed + stable_integer(class_name))
    return sorted(rng.sample(list(images), sample_count))


def resolution_scale_from_normalized_severity(
    norm_severity: float,
    min_scale: float,
    gamma: float
) -> float:
    """
    Normalized resolution mapping:

        alpha = severity / 100
        scale(alpha) = 1 - (1 - min_scale) * alpha^gamma

    With the defaults:
        severity 0   -> scale 1.00
        severity 100 -> scale 0.03
    """
    norm_severity = float(np.clip(norm_severity, 0.0, 1.0))

    if not 0.0 < min_scale <= 1.0:
        raise ValueError("min_scale must be greater than 0 and at most 1.")

    if gamma <= 0:
        raise ValueError("gamma must be greater than zero.")

    return 1.0 - (1.0 - min_scale) * (norm_severity ** gamma)


def apply_resolution_degradation(
    image: np.ndarray,
    scale: float,
    upsample_interpolation: int
):
    if scale >= 1.0:
        return image.copy(), image.shape[1], image.shape[0]

    original_height, original_width = image.shape[:2]

    reduced_width = max(1, int(round(original_width * scale)))
    reduced_height = max(1, int(round(original_height * scale)))

    reduced = cv2.resize(
        image,
        (reduced_width, reduced_height),
        interpolation=cv2.INTER_AREA
    )

    restored = cv2.resize(
        reduced,
        (original_width, original_height),
        interpolation=upsample_interpolation
    )

    return restored, reduced_width, reduced_height


def read_image(path: Path):
    try:
        encoded_data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(encoded_data, cv2.IMREAD_COLOR)
    except (OSError, ValueError):
        return None


def write_image(path: Path, image: np.ndarray) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    extension = path.suffix.lower()

    encoding_options = {
        ".jpg": (".jpg", [int(cv2.IMWRITE_JPEG_QUALITY), 95]),
        ".jpeg": (".jpg", [int(cv2.IMWRITE_JPEG_QUALITY), 95]),
        ".png": (".png", [int(cv2.IMWRITE_PNG_COMPRESSION), 3]),
        ".bmp": (".bmp", []),
        ".tif": (".tiff", []),
        ".tiff": (".tiff", []),
        ".webp": (".webp", [int(cv2.IMWRITE_WEBP_QUALITY), 95]),
    }

    encode_extension, parameters = encoding_options.get(
        extension,
        (".jpg", [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    )

    success, encoded_image = cv2.imencode(
        encode_extension,
        image,
        parameters
    )

    if not success:
        return False

    try:
        encoded_image.tofile(str(path))
        return True
    except OSError:
        return False


def scale_for_filename(scale: float) -> str:
    return f"{scale:.4f}".replace(".", "p")


def build_output_filename(
    source_path: Path,
    severity: int,
    scale: float
) -> str:
    return (
        f"{source_path.stem}"
        f"_resolution"
        f"_s{severity:03d}"
        f"_Sc{scale_for_filename(scale)}"
        f"{source_path.suffix}"
    )


def write_csv(path: Path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()

    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not input_root.exists():
        raise FileNotFoundError(
            f"Input root does not exist: {input_root}"
        )

    if not input_root.is_dir():
        raise NotADirectoryError(
            f"Input root is not a directory: {input_root}"
        )

    if args.minimum_per_class < 1:
        raise ValueError("minimum_per_class must be at least 1.")

    if not 0.0 < args.sampling_fraction <= 1.0:
        raise ValueError(
            "sampling_fraction must be greater than 0 and at most 1."
        )

    if not 0.0 < args.min_scale <= 1.0:
        raise ValueError("min_scale must be greater than 0 and at most 1.")

    if args.gamma <= 0:
        raise ValueError("gamma must be greater than zero.")

    output_root.mkdir(parents=True, exist_ok=True)

    severity_levels = parse_severity_levels(args.severity_levels)
    class_directories = find_class_directories(input_root)

    if not class_directories:
        raise RuntimeError(
            f"No class directories were found under {input_root}"
        )

    upsample_interpolation = INTERPOLATION_METHODS[
        args.upsample_method
    ]

    metadata_rows = []
    selection_rows = []

    total_available_sources = 0
    total_selected_sources = 0
    total_generated_images = 0
    total_skipped_existing = 0
    unreadable_images = 0
    failed_writes = 0

    print("=" * 72)
    print("Normalized Resolution-Degradation Generator")
    print("=" * 72)
    print(f"Input root:          {input_root}")
    print(f"Output root:         {output_root}")
    print(f"Minimum per class:   {args.minimum_per_class}")
    print(f"Sampling fraction:   {args.sampling_fraction:.4f}")
    print(f"Severity levels:     {severity_levels}")
    print(f"Minimum scale:       {args.min_scale:.4f}")
    print(f"Gamma:               {args.gamma:.4f}")
    print(f"Upsample method:     {args.upsample_method}")
    print(f"Random seed:         {args.seed}")
    print("=" * 72)

    for class_directory in class_directories:
        class_name = class_directory.name
        available_images = find_images(class_directory)

        if not available_images:
            print(f"[SKIP] {class_name}: no supported images")
            continue

        selected_images = select_weighted_images(
            images=available_images,
            class_name=class_name,
            minimum_per_class=args.minimum_per_class,
            sampling_fraction=args.sampling_fraction,
            seed=args.seed
        )

        total_available_sources += len(available_images)
        total_selected_sources += len(selected_images)

        output_class_directory = output_root / class_name
        output_class_directory.mkdir(parents=True, exist_ok=True)

        print(
            f"[CLASS] {class_name}: "
            f"{len(selected_images)} selected from "
            f"{len(available_images)}"
        )

        for source_index, source_path in enumerate(
            selected_images,
            start=1
        ):
            selection_rows.append({
                "true_label": class_name,
                "source_image_name": source_path.name,
                "source_image_path": str(source_path),
                "class_available_count": len(available_images),
                "class_selected_count": len(selected_images),
                "sampling_fraction": args.sampling_fraction,
                "minimum_per_class": args.minimum_per_class,
                "selection_seed": args.seed
            })

            image = read_image(source_path)

            if image is None:
                unreadable_images += 1
                print(f"  [READ ERROR] {source_path}")
                continue

            original_height, original_width = image.shape[:2]

            for severity in severity_levels:
                norm_severity = normalized_severity(severity)

                scale = resolution_scale_from_normalized_severity(
                    norm_severity=norm_severity,
                    min_scale=args.min_scale,
                    gamma=args.gamma
                )

                output_filename = build_output_filename(
                    source_path=source_path,
                    severity=severity,
                    scale=scale
                )

                output_path = output_class_directory / output_filename

                if output_path.exists() and not args.overwrite:
                    total_skipped_existing += 1
                    continue

                corrupted_image, reduced_width, reduced_height = (
                    apply_resolution_degradation(
                        image=image,
                        scale=scale,
                        upsample_interpolation=upsample_interpolation
                    )
                )

                if not write_image(output_path, corrupted_image):
                    failed_writes += 1
                    print(f"  [WRITE ERROR] {output_path}")
                    continue

                metadata_rows.append({
                    "true_label": class_name,
                    "source_image_name": source_path.name,
                    "source_image_path": str(source_path),
                    "generated_image_name": output_filename,
                    "generated_image_path": str(output_path),
                    "corruption_type": "resolution",
                    "severity": severity,
                    "normalized_severity": f"{norm_severity:.4f}",
                    "scale_factor": f"{scale:.6f}",
                    "original_width": original_width,
                    "original_height": original_height,
                    "reduced_width": reduced_width,
                    "reduced_height": reduced_height,
                    "downsample_method": "area",
                    "upsample_method": args.upsample_method,
                    "gamma": args.gamma,
                    "minimum_scale": args.min_scale,
                    "class_available_count": len(available_images),
                    "class_selected_count": len(selected_images),
                    "sampling_fraction": args.sampling_fraction,
                    "minimum_per_class": args.minimum_per_class,
                    "selection_seed": args.seed
                })

                total_generated_images += 1

            if source_index % 25 == 0:
                print(
                    f"  Processed {source_index}/"
                    f"{len(selected_images)} source images"
                )

    metadata_path = output_root / args.metadata_csv
    selection_path = output_root / args.selection_csv

    metadata_fieldnames = [
        "true_label",
        "source_image_name",
        "source_image_path",
        "generated_image_name",
        "generated_image_path",
        "corruption_type",
        "severity",
        "normalized_severity",
        "scale_factor",
        "original_width",
        "original_height",
        "reduced_width",
        "reduced_height",
        "downsample_method",
        "upsample_method",
        "gamma",
        "minimum_scale",
        "class_available_count",
        "class_selected_count",
        "sampling_fraction",
        "minimum_per_class",
        "selection_seed"
    ]

    selection_fieldnames = [
        "true_label",
        "source_image_name",
        "source_image_path",
        "class_available_count",
        "class_selected_count",
        "sampling_fraction",
        "minimum_per_class",
        "selection_seed"
    ]

    write_csv(metadata_path, metadata_rows, metadata_fieldnames)
    write_csv(selection_path, selection_rows, selection_fieldnames)

    expected_images = total_selected_sources * len(severity_levels)

    print()
    print("=" * 72)
    print("Generation complete")
    print("=" * 72)
    print(f"Available source images: {total_available_sources}")
    print(f"Selected source images:  {total_selected_sources}")
    print(f"Severity levels:         {len(severity_levels)}")
    print(f"Expected outputs:        {expected_images}")
    print(f"Generated outputs:       {total_generated_images}")
    print(f"Skipped existing:        {total_skipped_existing}")
    print(f"Unreadable sources:      {unreadable_images}")
    print(f"Failed image writes:     {failed_writes}")
    print(f"Metadata CSV:            {metadata_path}")
    print(f"Selection CSV:           {selection_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()