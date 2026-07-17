#!/usr/bin/env python3
"""
Normalized lighting-variation generator.

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
6. Uses the established nonlinear spatial lighting target:
       P(x,y) = 0.5 * (x/(W-1) + y/(H-1))
       B(x,y) = 1 / (1 + exp(-q(P-0.5)))
       T(x,y) = M_min + (M_max-M_min)B(x,y)
       M_s(x,y) = (1-alpha) + alpha*T(x,y)
7. Writes both a generation metadata CSV and a selected-sources CSV.
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


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate normalized lighting-variation images using "
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
        help="Output root for generated lighting-variation images."
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
        "--min_multiplier",
        type=float,
        default=0.0,
        help="Dark-end lighting multiplier at maximum severity."
    )
    parser.add_argument(
        "--max_multiplier",
        type=float,
        default=6.0,
        help="Bright-end lighting multiplier at maximum severity."
    )
    parser.add_argument(
        "--sigmoid_steepness",
        type=float,
        default=12.0,
        help="Steepness q of the spatial sigmoid lighting transition."
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help=(
            "Exponent applied to normalized severity before blending. "
            "The default 1.0 gives the established linear severity blend."
        )
    )
    parser.add_argument(
        "--gradient_direction",
        choices=[
            "top_left_to_bottom_right",
            "top_right_to_bottom_left",
            "bottom_left_to_top_right",
            "bottom_right_to_top_left"
        ],
        default="top_left_to_bottom_right",
        help="Direction from the darker side toward the brighter side."
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
        default="lighting_metadata.csv",
        help="Generation metadata CSV filename."
    )
    parser.add_argument(
        "--selection_csv",
        type=str,
        default="lighting_selected_sources.csv",
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


def create_spatial_progression(
    height: int,
    width: int,
    direction: str
) -> np.ndarray:
    """
    Build P(x,y), ranging approximately from 0 to 1 across the image.

        P(x,y) = 0.5 * (x/(W-1) + y/(H-1))
    """
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)

    xx, yy = np.meshgrid(x, y)

    if direction == "top_left_to_bottom_right":
        x_component = xx
        y_component = yy
    elif direction == "top_right_to_bottom_left":
        x_component = 1.0 - xx
        y_component = yy
    elif direction == "bottom_left_to_top_right":
        x_component = xx
        y_component = 1.0 - yy
    elif direction == "bottom_right_to_top_left":
        x_component = 1.0 - xx
        y_component = 1.0 - yy
    else:
        raise ValueError(f"Unsupported gradient direction: {direction}")

    return 0.5 * (x_component + y_component)


def create_target_lighting_map(
    height: int,
    width: int,
    min_multiplier: float,
    max_multiplier: float,
    sigmoid_steepness: float,
    gradient_direction: str
) -> np.ndarray:
    """
    Established target lighting field:

        P(x,y) = 0.5 * (x/(W-1) + y/(H-1))
        B(x,y) = 1 / (1 + exp(-q(P-0.5)))
        T(x,y) = M_min + (M_max-M_min)B(x,y)
    """
    if min_multiplier < 0:
        raise ValueError("min_multiplier cannot be negative.")

    if max_multiplier <= min_multiplier:
        raise ValueError(
            "max_multiplier must be greater than min_multiplier."
        )

    if sigmoid_steepness <= 0:
        raise ValueError("sigmoid_steepness must be greater than zero.")

    progression = create_spatial_progression(
        height=height,
        width=width,
        direction=gradient_direction
    )

    sigmoid_field = 1.0 / (
        1.0
        + np.exp(
            -sigmoid_steepness * (progression - 0.5)
        )
    )

    return (
        min_multiplier
        + (max_multiplier - min_multiplier) * sigmoid_field
    ).astype(np.float32)


def apply_lighting_variation(
    image: np.ndarray,
    norm_severity: float,
    target_lighting_map: np.ndarray,
    gamma: float
):
    """
    Blend the identity multiplier with the target lighting field:

        alpha = normalized_severity^gamma
        M_s(x,y) = (1-alpha) + alpha*T(x,y)
        I_s(x,y) = clip(I(x,y)*M_s(x,y), 0, 255)
    """
    norm_severity = float(np.clip(norm_severity, 0.0, 1.0))

    if gamma <= 0:
        raise ValueError("gamma must be greater than zero.")

    alpha = norm_severity ** gamma

    if alpha == 0.0:
        identity_map = np.ones_like(
            target_lighting_map,
            dtype=np.float32
        )
        return image.copy(), identity_map, alpha

    severity_map = (
        (1.0 - alpha)
        + alpha * target_lighting_map
    ).astype(np.float32)

    image_float = image.astype(np.float32)
    corrupted = image_float * severity_map[:, :, None]
    corrupted = np.clip(corrupted, 0.0, 255.0).astype(np.uint8)

    return corrupted, severity_map, alpha


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


def build_output_filename(
    source_path: Path,
    severity: int
) -> str:
    return (
        f"{source_path.stem}"
        f"_lighting"
        f"_s{severity:03d}"
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

    if args.min_multiplier < 0:
        raise ValueError("min_multiplier cannot be negative.")

    if args.max_multiplier <= args.min_multiplier:
        raise ValueError(
            "max_multiplier must be greater than min_multiplier."
        )

    if args.sigmoid_steepness <= 0:
        raise ValueError(
            "sigmoid_steepness must be greater than zero."
        )

    if args.gamma <= 0:
        raise ValueError("gamma must be greater than zero.")

    output_root.mkdir(parents=True, exist_ok=True)

    severity_levels = parse_severity_levels(args.severity_levels)
    class_directories = find_class_directories(input_root)

    if not class_directories:
        raise RuntimeError(
            f"No class directories were found under {input_root}"
        )

    metadata_rows = []
    selection_rows = []

    total_available_sources = 0
    total_selected_sources = 0
    total_generated_images = 0
    total_skipped_existing = 0
    unreadable_images = 0
    failed_writes = 0

    print("=" * 72)
    print("Normalized Lighting-Variation Generator")
    print("=" * 72)
    print(f"Input root:          {input_root}")
    print(f"Output root:         {output_root}")
    print(f"Minimum per class:   {args.minimum_per_class}")
    print(f"Sampling fraction:   {args.sampling_fraction:.4f}")
    print(f"Severity levels:     {severity_levels}")
    print(f"Minimum multiplier:  {args.min_multiplier:.4f}")
    print(f"Maximum multiplier:  {args.max_multiplier:.4f}")
    print(f"Sigmoid steepness:   {args.sigmoid_steepness:.4f}")
    print(f"Gamma:               {args.gamma:.4f}")
    print(f"Gradient direction:  {args.gradient_direction}")
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

            height, width = image.shape[:2]

            target_lighting_map = create_target_lighting_map(
                height=height,
                width=width,
                min_multiplier=args.min_multiplier,
                max_multiplier=args.max_multiplier,
                sigmoid_steepness=args.sigmoid_steepness,
                gradient_direction=args.gradient_direction
            )

            target_map_min = float(target_lighting_map.min())
            target_map_max = float(target_lighting_map.max())

            for severity in severity_levels:
                norm_severity = normalized_severity(severity)

                output_filename = build_output_filename(
                    source_path=source_path,
                    severity=severity
                )

                output_path = output_class_directory / output_filename

                if output_path.exists() and not args.overwrite:
                    total_skipped_existing += 1
                    continue

                corrupted_image, severity_map, alpha = (
                    apply_lighting_variation(
                        image=image,
                        norm_severity=norm_severity,
                        target_lighting_map=target_lighting_map,
                        gamma=args.gamma
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
                    "corruption_type": "lighting",
                    "severity": severity,
                    "normalized_severity": f"{norm_severity:.4f}",
                    "severity_alpha": f"{alpha:.6f}",
                    "applied_multiplier_min": (
                        f"{float(severity_map.min()):.6f}"
                    ),
                    "applied_multiplier_max": (
                        f"{float(severity_map.max()):.6f}"
                    ),
                    "target_multiplier_min": f"{target_map_min:.6f}",
                    "target_multiplier_max": f"{target_map_max:.6f}",
                    "configured_min_multiplier": args.min_multiplier,
                    "configured_max_multiplier": args.max_multiplier,
                    "sigmoid_steepness": args.sigmoid_steepness,
                    "gamma": args.gamma,
                    "gradient_direction": args.gradient_direction,
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
        "severity_alpha",
        "applied_multiplier_min",
        "applied_multiplier_max",
        "target_multiplier_min",
        "target_multiplier_max",
        "configured_min_multiplier",
        "configured_max_multiplier",
        "sigmoid_steepness",
        "gamma",
        "gradient_direction",
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