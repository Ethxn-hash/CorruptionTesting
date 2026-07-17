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
            "Generate normalized motion-blur corruption images using "
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
        help="Output root for generated motion-blur images."
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
        help=(
            "Fraction of each class selected. The final count is at least "
            "minimum_per_class when enough images are available."
        )
    )

    parser.add_argument(
        "--severity_levels",
        type=str,
        default="0,10,20,30,40,50,60,70,80,90,100",
        help="Comma-separated normalized severity levels from 0 to 100."
    )

    parser.add_argument(
        "--min_kernel",
        type=int,
        default=1,
        help="Motion-blur kernel size at normalized severity 0."
    )

    parser.add_argument(
        "--max_kernel",
        type=int,
        default=51,
        help="Motion-blur kernel size at normalized severity 100."
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Exponent controlling the normalized-severity mapping."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed for reproducible image selection and blur angles."
    )

    parser.add_argument(
        "--metadata_csv",
        type=str,
        default="motion_blur_metadata.csv",
        help="Metadata CSV filename."
    )

    parser.add_argument(
        "--selection_csv",
        type=str,
        default="motion_blur_selected_sources.csv",
        help="CSV recording which original images were selected."
    )

    return parser.parse_args()


def stable_integer(text: str) -> int:
    """Create a stable integer from text for reproducible randomization."""
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
                f"Severity {severity} is outside the valid range 0–100."
            )

    return levels


def normalized_severity(severity: int) -> float:
    """Convert severity 0–100 into normalized severity 0.0–1.0."""
    return float(np.clip(severity, 0, 100)) / 100.0


def calculate_sample_count(
    class_size: int,
    minimum_per_class: int,
    sampling_fraction: float
) -> int:
    """
    Weighted class sampling rule:

        selected_count =
            min(class_size,
                max(minimum_per_class,
                    round(class_size * sampling_fraction)))

    Classes with fewer than the requested minimum use all available images.
    """

    if class_size <= 0:
        return 0

    proportional_count = round(class_size * sampling_fraction)
    requested_count = max(minimum_per_class, proportional_count)

    return min(class_size, requested_count)


def find_class_directories(input_root: Path):
    return sorted(
        path
        for path in input_root.iterdir()
        if path.is_dir()
    )


def find_images(class_directory: Path):
    return sorted(
        path
        for path in class_directory.iterdir()
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

    class_seed = seed + stable_integer(class_name)
    rng = random.Random(class_seed)

    selected = rng.sample(list(images), sample_count)
    return sorted(selected)


def ensure_odd(value: int) -> int:
    value = max(1, int(round(value)))

    if value % 2 == 0:
        value += 1

    return value


def kernel_size_from_normalized_severity(
    norm_severity: float,
    min_kernel: int,
    max_kernel: int,
    gamma: float
) -> int:
    """
    Convert normalized severity into motion-blur kernel size.

        scaled severity = normalized_severity ** gamma

        kernel =
            min_kernel
            + scaled_severity * (max_kernel - min_kernel)
    """

    norm_severity = float(np.clip(norm_severity, 0.0, 1.0))

    min_kernel = ensure_odd(min_kernel)
    max_kernel = ensure_odd(max_kernel)

    if max_kernel < min_kernel:
        raise ValueError("max_kernel must be greater than or equal to min_kernel.")

    if gamma <= 0:
        raise ValueError("gamma must be greater than zero.")

    if norm_severity == 0.0:
        return 1

    scaled_severity = norm_severity ** gamma

    raw_kernel = (
        min_kernel
        + scaled_severity * (max_kernel - min_kernel)
    )

    return ensure_odd(raw_kernel)


def deterministic_blur_angle(
    source_path: Path,
    input_root: Path,
    seed: int
) -> float:
    """
    Give each source image one reproducible motion direction.

    The same image keeps the same angle across all severity levels.
    """

    relative_path = source_path.relative_to(input_root).as_posix()
    image_seed = seed + stable_integer(relative_path)

    rng = random.Random(image_seed)
    return rng.uniform(0.0, 180.0)


def create_motion_blur_kernel(
    kernel_size: int,
    angle_degrees: float
) -> np.ndarray:
    kernel_size = ensure_odd(kernel_size)

    kernel = np.zeros(
        (kernel_size, kernel_size),
        dtype=np.float32
    )

    center_index = kernel_size // 2
    kernel[center_index, :] = 1.0

    center = (
        (kernel_size - 1) / 2.0,
        (kernel_size - 1) / 2.0
    )

    rotation_matrix = cv2.getRotationMatrix2D(
        center,
        angle_degrees,
        1.0
    )

    rotated_kernel = cv2.warpAffine(
        kernel,
        rotation_matrix,
        (kernel_size, kernel_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    kernel_sum = float(rotated_kernel.sum())

    if kernel_sum <= 0:
        rotated_kernel[center_index, center_index] = 1.0
        kernel_sum = 1.0

    return rotated_kernel / kernel_sum


def apply_motion_blur(
    image: np.ndarray,
    kernel_size: int,
    angle_degrees: float
) -> np.ndarray:
    if kernel_size <= 1:
        return image.copy()

    kernel = create_motion_blur_kernel(
        kernel_size=kernel_size,
        angle_degrees=angle_degrees
    )

    return cv2.filter2D(
        image,
        ddepth=-1,
        kernel=kernel,
        borderType=cv2.BORDER_REFLECT
    )


def read_image(path: Path):
    """
    Read images through imdecode so paths containing spaces or
    non-ASCII characters work correctly.
    """

    try:
        encoded_data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(encoded_data, cv2.IMREAD_COLOR)
    except (OSError, ValueError):
        return None


def write_image(path: Path, image: np.ndarray) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)

    extension = path.suffix.lower()

    encoding_options = {
        ".jpg": (
            ".jpg",
            [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        ),
        ".jpeg": (
            ".jpg",
            [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        ),
        ".png": (
            ".png",
            [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
        ),
        ".bmp": (
            ".bmp",
            []
        ),
        ".tif": (
            ".tiff",
            []
        ),
        ".tiff": (
            ".tiff",
            []
        ),
        ".webp": (
            ".webp",
            [int(cv2.IMWRITE_WEBP_QUALITY), 95]
        )
    }

    encode_extension, parameters = encoding_options.get(
        extension,
        (
            ".jpg",
            [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        )
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
        f"_motion_blur"
        f"_s{severity:03d}"
        f"{source_path.suffix}"
    )


def write_csv(path: Path, rows, fieldnames):
    with path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

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

    if not 0 < args.sampling_fraction <= 1:
        raise ValueError(
            "sampling_fraction must be greater than 0 and no greater than 1."
        )

    output_root.mkdir(parents=True, exist_ok=True)

    severity_levels = parse_severity_levels(
        args.severity_levels
    )

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
    unreadable_images = 0
    failed_writes = 0

    print("=" * 70)
    print("Normalized Motion Blur Generator")
    print("=" * 70)
    print(f"Input root:          {input_root}")
    print(f"Output root:         {output_root}")
    print(f"Minimum per class:   {args.minimum_per_class}")
    print(f"Sampling fraction:   {args.sampling_fraction:.4f}")
    print(f"Severity levels:     {severity_levels}")
    print(f"Random seed:         {args.seed}")
    print("=" * 70)

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
        output_class_directory.mkdir(
            parents=True,
            exist_ok=True
        )

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

            angle_degrees = deterministic_blur_angle(
                source_path=source_path,
                input_root=input_root,
                seed=args.seed
            )

            for severity in severity_levels:
                norm_severity = normalized_severity(severity)

                kernel_size = kernel_size_from_normalized_severity(
                    norm_severity=norm_severity,
                    min_kernel=args.min_kernel,
                    max_kernel=args.max_kernel,
                    gamma=args.gamma
                )

                if norm_severity == 0.0:
                    corrupted_image = image.copy()
                else:
                    corrupted_image = apply_motion_blur(
                        image=image,
                        kernel_size=kernel_size,
                        angle_degrees=angle_degrees
                    )

                output_filename = build_output_filename(
                    source_path=source_path,
                    severity=severity
                )

                output_path = (
                    output_class_directory
                    / output_filename
                )

                successfully_written = write_image(
                    output_path,
                    corrupted_image
                )

                if not successfully_written:
                    failed_writes += 1
                    print(f"  [WRITE ERROR] {output_path}")
                    continue

                metadata_rows.append({
                    "true_label": class_name,
                    "source_image_name": source_path.name,
                    "source_image_path": str(source_path),
                    "generated_image_name": output_filename,
                    "generated_image_path": str(output_path),
                    "corruption_type": "motion_blur",
                    "severity": severity,
                    "normalized_severity": f"{norm_severity:.4f}",
                    "kernel_size": kernel_size,
                    "angle_degrees": f"{angle_degrees:.4f}",
                    "gamma": args.gamma,
                    "min_kernel": args.min_kernel,
                    "max_kernel": args.max_kernel,
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
        "kernel_size",
        "angle_degrees",
        "gamma",
        "min_kernel",
        "max_kernel",
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

    write_csv(
        metadata_path,
        metadata_rows,
        metadata_fieldnames
    )

    write_csv(
        selection_path,
        selection_rows,
        selection_fieldnames
    )

    expected_images = (
        total_selected_sources
        * len(severity_levels)
    )

    print()
    print("=" * 70)
    print("Generation complete")
    print("=" * 70)
    print(f"Available source images: {total_available_sources}")
    print(f"Selected source images:  {total_selected_sources}")
    print(f"Severity levels:         {len(severity_levels)}")
    print(f"Expected outputs:        {expected_images}")
    print(f"Generated outputs:       {total_generated_images}")
    print(f"Unreadable sources:      {unreadable_images}")
    print(f"Failed image writes:     {failed_writes}")
    print(f"Metadata CSV:            {metadata_path}")
    print(f"Selection CSV:           {selection_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()