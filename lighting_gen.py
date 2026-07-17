from __future__ import annotations
import argparse
import csv
import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

import normalized_severity as ns


SEVERITIES = list(range(0, 101, 10))
MIN_IMAGES_PER_CLASS = 100
CLASS_FRACTION = 0.086
RANDOM_SEED = 2026
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp",
}

MANIFEST_FIELDS = [
    "selection_index",
    "class_path",
    "source_relative_path",
    "class_total_images",
    "selected_from_class",
    "selected_fraction",
    "random_seed",
]


def is_inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def target_count(class_total: int) -> int:
    """
    Exact sampling rule:
        min(N_c, max(100, round(0.086 * N_c)))
    """
    return min(
        class_total,
        max(
            MIN_IMAGES_PER_CLASS,
            int(round(CLASS_FRACTION * class_total)),
        ),
    )


def discover_by_class(
    input_root: Path,
    excluded_root: Path | None = None,
) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)

    for path in input_root.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        if excluded_root is not None and is_inside(path, excluded_root):
            continue

        relative = path.relative_to(input_root)
        class_path = relative.parent.as_posix()
        grouped[class_path].append(path)

    for class_path in grouped:
        grouped[class_path].sort(
            key=lambda p: p.relative_to(input_root).as_posix().lower()
        )

    return dict(grouped)


def create_manifest(
    input_root: Path,
    manifest_path: Path,
    excluded_root: Path | None = None,
) -> list[dict[str, str]]:
    grouped = discover_by_class(input_root, excluded_root)

    if not grouped:
        raise FileNotFoundError(
            f"No supported images found under: {input_root}"
        )

    rng = random.Random(RANDOM_SEED)
    rows: list[dict[str, str]] = []
    selection_index = 0

    for class_path in sorted(grouped, key=str.lower):
        images = grouped[class_path]
        class_total = len(images)
        selected_count = target_count(class_total)

        if selected_count >= class_total:
            selected = list(images)
        else:
            selected = rng.sample(images, selected_count)
            selected.sort(
                key=lambda p: p.relative_to(input_root).as_posix().lower()
            )

        fraction = selected_count / class_total

        for image_path in selected:
            rows.append(
                {
                    "selection_index": str(selection_index),
                    "class_path": class_path,
                    "source_relative_path": (
                        image_path.relative_to(input_root).as_posix()
                    ),
                    "class_total_images": str(class_total),
                    "selected_from_class": str(selected_count),
                    "selected_fraction": f"{fraction:.10f}",
                    "random_seed": str(RANDOM_SEED),
                }
            )
            selection_index += 1

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = manifest_path.with_suffix(
        manifest_path.suffix + ".tmp"
    )

    with temp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    temp_path.replace(manifest_path)

    print(f"Created shared manifest: {manifest_path}")
    print(f"Selected images: {len(rows)}")
    print(f"Classes: {len(grouped)}")
    print("Rule: min(N_c, max(100, round(0.086 * N_c)))")
    print(f"Seed: {RANDOM_SEED}")

    return rows


def load_manifest(
    input_root: Path,
    manifest_path: Path,
) -> list[dict[str, str]]:
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []

        missing = [
            field for field in MANIFEST_FIELDS
            if field not in fields
        ]

        if missing:
            raise ValueError(
                f"Manifest is missing required columns: {missing}"
            )

        rows = list(reader)

    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")

    seen: set[str] = set()

    for row_number, row in enumerate(rows, start=2):
        relative_text = row["source_relative_path"]
        relative_path = Path(relative_text)

        if relative_path.is_absolute():
            raise ValueError(
                f"Absolute path in manifest row {row_number}: "
                f"{relative_text}"
            )

        source_path = (input_root / relative_path).resolve()

        if not is_inside(source_path, input_root):
            raise ValueError(
                f"Manifest path escapes input root: {relative_text}"
            )

        if not source_path.is_file():
            raise FileNotFoundError(
                f"Manifest source does not exist: {source_path}"
            )

        if relative_text in seen:
            raise ValueError(
                f"Duplicate manifest entry: {relative_text}"
            )

        seen.add(relative_text)

    print(f"Loaded shared manifest: {manifest_path}")
    print(f"Selected images: {len(rows)}")
    return rows


def get_manifest(
    input_root: Path,
    output_root: Path,
    manifest_arg: Path | None,
    rebuild: bool,
) -> tuple[Path, list[dict[str, str]]]:
    if manifest_arg is None:
        manifest_path = (
            input_root.parent
            / "plantvillage_selected_images.csv"
        )
    else:
        manifest_path = manifest_arg.expanduser().resolve()

    if rebuild or not manifest_path.exists():
        rows = create_manifest(
            input_root,
            manifest_path,
            excluded_root=output_root,
        )
    else:
        rows = load_manifest(input_root, manifest_path)

    return manifest_path, rows


def build_output_stems(
    manifest_rows: list[dict[str, str]],
) -> dict[str, str]:
    counts: Counter[tuple[str, str]] = Counter()

    for row in manifest_rows:
        relative = Path(row["source_relative_path"])
        counts[
            (row["class_path"], relative.stem.casefold())
        ] += 1

    output_stems: dict[str, str] = {}

    for row in manifest_rows:
        relative_text = row["source_relative_path"]
        relative = Path(relative_text)
        key = (row["class_path"], relative.stem.casefold())
        stem = relative.stem

        if counts[key] > 1:
            suffix = hashlib.sha1(
                relative_text.encode("utf-8")
            ).hexdigest()[:8]
            stem = f"{stem}__{suffix}"

        output_stems[relative_text] = stem

    return output_stems


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Could not read image: {path}")

    return image


def output_dir(
    root: Path,
    severity: int,
    class_path: str,
) -> Path:
    directory = root / f"severity_{severity:03d}"

    if class_path not in {"", "."}:
        directory = directory / Path(class_path)

    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_png(path: Path, image: np.ndarray) -> None:
    success = cv2.imwrite(
        str(path),
        image,
        [cv2.IMWRITE_PNG_COMPRESSION, 3],
    )

    if not success:
        raise IOError(f"Failed to save image: {path}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--rebuild_manifest", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress_every", type=int, default=100)
    return parser.parse_args()


def ensure_bgr(image: np.ndarray) -> np.ndarray:
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


def odd(value: float) -> int:
    value = max(3, int(round(value)))
    return value if value % 2 == 1 else value + 1


def make_lighting_map(
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, int, float, float]:
    image = ensure_bgr(image)
    minimum_dimension = min(image.shape[:2])

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    luminance = lab[:, :, 0].astype(np.float32) / 255.0

    sigma = float(
        np.clip(
            minimum_dimension * 0.035,
            3.0,
            35.0,
        )
    )
    kernel_size = odd(6.0 * sigma + 1.0)

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
        normalized = np.full_like(
            smoothed,
            0.5,
            dtype=np.float32,
        )
    else:
        normalized = np.clip(
            (smoothed - low) / (high - low),
            0.0,
            1.0,
        ).astype(np.float32)

    signed = (2.0 * normalized - 1.0).astype(np.float32)

    return normalized, signed, sigma, kernel_size, low, high


def lighting_parameters(
    severity: int,
) -> tuple[float, float]:
    if severity == 0:
        return 0.0, 0.0

    push = float(
        np.clip(
            ns.get_lighting_push(severity),
            0.0,
            1.5,
        )
    )
    blend = float(
        np.clip(
            ns.get_lighting_bw_blend(severity),
            0.0,
            1.0,
        )
    )

    return push, blend


def apply_lighting(
    image: np.ndarray,
    severity: int,
    normalized: np.ndarray,
    signed: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    image = ensure_bgr(image)
    push, blend = lighting_parameters(severity)

    if severity == 0:
        return image.copy(), push, blend

    image_float = image.astype(np.float32)
    strength = (np.abs(signed) ** 0.55).astype(np.float32)

    signed_3 = signed[:, :, None]
    strength_3 = strength[:, :, None]

    bright_weight = np.clip(
        signed_3,
        0.0,
        1.0,
    ) * strength_3

    dark_weight = np.clip(
        -signed_3,
        0.0,
        1.0,
    ) * strength_3

    illuminated = (
        image_float
        + push * bright_weight * (255.0 - image_float)
        - push * dark_weight * image_float
    )
    illuminated = np.clip(illuminated, 0.0, 255.0)

    binary_gray = np.where(
        normalized >= 0.5,
        255.0,
        0.0,
    ).astype(np.float32)

    binary = np.repeat(
        binary_gray[:, :, None],
        3,
        axis=2,
    )

    corrupted = (
        (1.0 - blend) * illuminated
        + blend * binary
    )

    if severity == 100:
        corrupted = binary.copy()

    return (
        ensure_bgr(
            np.clip(corrupted, 0, 255).astype(np.uint8)
        ),
        push,
        blend,
    )


def main():
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not input_root.is_dir():
        raise NotADirectoryError(
            f"Input root does not exist or is not a directory: {input_root}"
        )

    output_root.mkdir(parents=True, exist_ok=True)

    manifest_path, rows = get_manifest(
        input_root,
        output_root,
        args.manifest,
        args.rebuild_manifest,
    )
    stems = build_output_stems(rows)

    metadata_path = output_root / "lighting_metadata.csv"
    written = skipped = failed = 0

    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "selection_index",
            "source_relative_path",
            "class_path",
            "class_total_images",
            "selected_from_class",
            "factor",
            "severity",
            "lighting_push",
            "black_white_blend",
            "map_sigma",
            "map_kernel_size",
            "percentile_low",
            "percentile_high",
            "output_relative_path",
            "status",
            "error",
            "shared_manifest",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        for index, row in enumerate(rows, start=1):
            relative_text = row["source_relative_path"]
            source_path = input_root / Path(relative_text)
            class_path = row["class_path"]

            try:
                image = read_image(source_path)
                (
                    normalized,
                    signed,
                    sigma,
                    kernel_size,
                    low,
                    high,
                ) = make_lighting_map(image)
            except Exception as error:
                failed += len(SEVERITIES)

                for severity in SEVERITIES:
                    writer.writerow({
                        "selection_index": row["selection_index"],
                        "source_relative_path": relative_text,
                        "class_path": class_path,
                        "class_total_images": row["class_total_images"],
                        "selected_from_class": row["selected_from_class"],
                        "factor": "lighting",
                        "severity": severity,
                        "lighting_push": "",
                        "black_white_blend": "",
                        "map_sigma": "",
                        "map_kernel_size": "",
                        "percentile_low": "",
                        "percentile_high": "",
                        "output_relative_path": "",
                        "status": "failed",
                        "error": str(error),
                        "shared_manifest": str(manifest_path),
                    })
                continue

            for severity in SEVERITIES:
                destination = output_dir(
                    output_root,
                    severity,
                    class_path,
                ) / (
                    f"{stems[relative_text]}_lighting_s{severity:03d}.png"
                )

                status = "written"
                error_text = ""
                push = blend = ""

                try:
                    push, blend = lighting_parameters(severity)

                    if destination.exists() and not args.overwrite:
                        status = "skipped_existing"
                        skipped += 1
                    else:
                        corrupted, push, blend = apply_lighting(
                            image,
                            severity,
                            normalized,
                            signed,
                        )
                        save_png(destination, corrupted)
                        written += 1

                except Exception as error:
                    status = "failed"
                    error_text = str(error)
                    failed += 1

                writer.writerow({
                    "selection_index": row["selection_index"],
                    "source_relative_path": relative_text,
                    "class_path": class_path,
                    "class_total_images": row["class_total_images"],
                    "selected_from_class": row["selected_from_class"],
                    "factor": "lighting",
                    "severity": severity,
                    "lighting_push": push,
                    "black_white_blend": blend,
                    "map_sigma": sigma,
                    "map_kernel_size": kernel_size,
                    "percentile_low": low,
                    "percentile_high": high,
                    "output_relative_path": (
                        destination.relative_to(output_root).as_posix()
                        if status != "failed"
                        else ""
                    ),
                    "status": status,
                    "error": error_text,
                    "shared_manifest": str(manifest_path),
                })

            if (
                args.progress_every > 0
                and (
                    index % args.progress_every == 0
                    or index == len(rows)
                )
            ):
                print(
                    f"[{index}/{len(rows)}] "
                    f"written={written}, skipped={skipped}, failed={failed}"
                )

    print("Lighting generation complete.")
    print(f"Manifest: {manifest_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Written={written}, skipped={skipped}, failed={failed}")


if __name__ == "__main__":
    main()