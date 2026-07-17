import argparse
import math
import os

import cv2
import numpy as np

import normalized_severity as ns


SEVERITIES = getattr(
    ns,
    "SEVERITIES",
    list(range(0, 101, 10)),
)


def read_image(path):
    """
    Read an image as a three-channel BGR image.
    """

    image = cv2.imread(path, cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {path}\n"
            "Check the path, filename, and extension."
        )

    return image


def ensure_directory(path):
    os.makedirs(path, exist_ok=True)


def ensure_bgr(image):
    """
    Guarantee that an image has exactly three BGR channels.
    """

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if image.ndim == 3:
        if image.shape[2] == 1:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        if image.shape[2] == 3:
            return image

    raise ValueError(
        f"Unsupported image shape: {image.shape}"
    )


def force_odd(value):
    """
    Convert a number to a positive odd integer.
    """

    value = max(3, int(round(value)))

    if value % 2 == 0:
        value += 1

    return value


def create_lighting_map(image):
    """
    Create a smooth illumination map using the image's existing luminance.

    Broad bright regions receive positive values.
    Broad dark regions receive negative values.

    Returns:
        normalized_map:
            Luminance values normalized to approximately 0–1.

        signed_map:
            Dark regions near -1 and bright regions near +1.
    """

    image = ensure_bgr(image)

    height, width = image.shape[:2]
    minimum_dimension = min(height, width)

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB,
    )

    luminance = (
        lab[:, :, 0].astype(np.float32) / 255.0
    )

    # Broad smoothing helps the corruption follow realistic illumination
    # regions rather than individual leaf edges or pixel noise.
    sigma = float(
        np.clip(
            minimum_dimension * 0.035,
            3.0,
            35.0,
        )
    )

    kernel_size = force_odd(
        6.0 * sigma + 1.0
    )

    smoothed = cv2.GaussianBlur(
        luminance,
        (kernel_size, kernel_size),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT101,
    )

    # Percentile normalization prevents isolated extreme pixels from
    # controlling the entire lighting map.
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
    Apply harsh illumination corruption.

    Behavior:
        Severity 0:
            Original image.

        Increasing severity:
            Existing bright regions move toward white.
            Existing dark regions move toward black.

        Severity 100:
            A pure black-and-white, three-channel BGR image.

    The parameter values are obtained from normalized_severity.py.
    """

    image = ensure_bgr(image)

    severity = float(
        np.clip(severity, 0, 100)
    )

    if severity <= 0:
        return image.copy(), 0.0, 0.0

    push = float(
        ns.get_lighting_push(severity)
    )

    bw_blend = float(
        ns.get_lighting_bw_blend(severity)
    )

    push = float(
        np.clip(push, 0.0, 1.5)
    )

    bw_blend = float(
        np.clip(bw_blend, 0.0, 1.0)
    )

    image_float = image.astype(np.float32)

    normalized_map, signed_map = (
        create_lighting_map(image)
    )

    # Strengthen middle-valued regions slightly so changes are visible
    # throughout the severity sequence.
    region_strength = (
        np.abs(signed_map) ** 0.55
    ).astype(np.float32)

    # Convert maps from H x W to H x W x 1.
    signed_map_3 = signed_map[:, :, None]
    strength_3 = region_strength[:, :, None]

    bright_weight = (
        np.clip(
            signed_map_3,
            0.0,
            1.0,
        )
        * strength_3
    )

    dark_weight = (
        np.clip(
            -signed_map_3,
            0.0,
            1.0,
        )
        * strength_3
    )

    # Bright regions move toward white.
    bright_change = (
        push
        * bright_weight
        * (255.0 - image_float)
    )

    # Dark regions move toward black.
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

    # Create a two-dimensional black-and-white target.
    binary_mask = normalized_map >= 0.5

    binary_gray = np.where(
        binary_mask,
        255.0,
        0.0,
    ).astype(np.float32)

    # Convert H x W grayscale target into H x W x 3.
    binary_target = np.repeat(
        binary_gray[:, :, None],
        3,
        axis=2,
    )

    # Gradually transition toward the black-and-white endpoint.
    corrupted = (
        (1.0 - bw_blend) * illumination_image
        + bw_blend * binary_target
    )

    if severity >= 100:
        # Guarantee severity 100 contains only black and white and
        # remains a three-channel BGR image.
        corrupted = binary_target.copy()

    corrupted = np.clip(
        corrupted,
        0.0,
        255.0,
    ).astype(np.uint8)

    corrupted = ensure_bgr(corrupted)

    return corrupted, push, bw_blend


def resize_for_cell(
    image,
    available_width,
    available_height,
):
    """
    Resize an image while preserving its aspect ratio.
    """

    image = ensure_bgr(image)

    height, width = image.shape[:2]

    scale = min(
        available_width / width,
        available_height / height,
    )

    new_width = max(
        1,
        int(round(width * scale)),
    )

    new_height = max(
        1,
        int(round(height * scale)),
    )

    interpolation = (
        cv2.INTER_AREA
        if scale < 1.0
        else cv2.INTER_NEAREST
    )

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=interpolation,
    )

    return ensure_bgr(resized)


def create_contact_sheet(
    labeled_images,
    columns=3,
    cell_width=300,
    cell_height=260,
):
    """
    Create a contact sheet from three-channel BGR images.
    """

    rows = math.ceil(
        len(labeled_images) / columns
    )

    sheet = np.full(
        (
            rows * cell_height,
            columns * cell_width,
            3,
        ),
        255,
        dtype=np.uint8,
    )

    for index, (label, image) in enumerate(
        labeled_images
    ):
        image = ensure_bgr(image)

        row = index // columns
        column = index % columns

        x_start = column * cell_width
        y_start = row * cell_height

        preview = resize_for_cell(
            image,
            available_width=cell_width - 20,
            available_height=cell_height - 55,
        )

        preview = ensure_bgr(preview)

        preview_height, preview_width = (
            preview.shape[:2]
        )

        x_offset = (
            x_start
            + (cell_width - preview_width) // 2
        )

        y_offset = (
            y_start
            + 42
            + (
                cell_height
                - 42
                - preview_height
            ) // 2
        )

        destination = sheet[
            y_offset:y_offset + preview_height,
            x_offset:x_offset + preview_width,
        ]

        if destination.shape != preview.shape:
            raise ValueError(
                "Contact-sheet shape mismatch:\n"
                f"Destination: {destination.shape}\n"
                f"Preview: {preview.shape}"
            )

        destination[:] = preview

        cv2.putText(
            sheet,
            label,
            (x_start + 8, y_start + 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return sheet


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate normalized harsh-illumination "
            "corruption at severities 0, 10, ..., 100."
        )
    )

    parser.add_argument(
        "image",
        help="Path to the input image.",
    )

    parser.add_argument(
        "--output_dir",
        default="lighting_test",
        help=(
            "Directory used to save generated images. "
            "Default: lighting_test"
        ),
    )

    args = parser.parse_args()

    image = read_image(args.image)
    ensure_directory(args.output_dir)

    base_name = os.path.splitext(
        os.path.basename(args.image)
    )[0]

    contact_images = []

    for severity in SEVERITIES:
        corrupted, push, bw_blend = (
            apply_lighting_degradation(
                image,
                severity,
            )
        )

        output_name = (
            f"{base_name}_lighting_"
            f"s{int(severity):03d}.png"
        )

        output_path = os.path.join(
            args.output_dir,
            output_name,
        )

        success = cv2.imwrite(
            output_path,
            corrupted,
        )

        if not success:
            raise IOError(
                f"Failed to save: {output_path}"
            )

        parameter_text = (
            f"push={push:.2f}, "
            f"BW={bw_blend:.2f}"
        )

        label = (
            f"Severity {int(severity)} | "
            f"{parameter_text}"
        )

        contact_images.append(
            (label, corrupted)
        )

        print(
            f"Saved severity {int(severity)}: "
            f"{output_path} "
            f"({parameter_text}, "
            f"shape={corrupted.shape})"
        )

    contact_sheet = create_contact_sheet(
        contact_images
    )

    contact_sheet_path = os.path.join(
        args.output_dir,
        f"{base_name}_lighting_contact_sheet.png",
    )

    success = cv2.imwrite(
        contact_sheet_path,
        contact_sheet,
    )

    if not success:
        raise IOError(
            f"Failed to save contact sheet: "
            f"{contact_sheet_path}"
        )

    print(
        f"\nContact sheet: "
        f"{contact_sheet_path}"
    )


if __name__ == "__main__":
    main()