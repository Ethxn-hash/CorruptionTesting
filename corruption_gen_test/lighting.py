#!/usr/bin/env python3
"""
Preview a smooth, normalized lighting-corruption sequence.

Generates:
    severity 0, 20, 50, 80, and 100
plus a labeled contact sheet.

The lighting mask is continuous across the entire image:
- no coverage floor,
- no np.where split,
- no binary black/white endpoint,
- no unused highlight/shadow parameters.

The values in the EDITABLE SETTINGS section directly control the pixels.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np


# =====================================================================
# EDITABLE SETTINGS
# Change values only in this section while calibrating the formula.
# =====================================================================

# Preview severity levels.
PREVIEW_SEVERITIES = (0, 20, 50, 80, 100)

# Broad spatial lighting pattern:
# "diagonal", "horizontal", "vertical", "radial", or "vignette"
PATTERN = "diagonal"

# Controls how rapidly corruption grows from severity 0 to 100.
# Higher values keep low severities milder while preserving severity 100.
# Suggested range: 1.4 to 2.2
SEVERITY_GAMMA = 2.00

# Maximum positive exposure at severity 100, in photographic stops.
# +1 stop = 2x brightness; +5 stops = 32x brightness before clipping.
# Increase this to blow out more of the bright side.
# Suggested range: 3.0 to 7.0
HIGHLIGHT_STOPS = 7.00

# Maximum negative exposure at severity 100, in photographic stops.
# -1 stop = 1/2 brightness; -6 stops = 1/64 brightness.
# Increase this to make the shadow side darker.
# Suggested range: 4.0 to 8.0
SHADOW_STOPS = 8.00

# Controls the smoothness of the light-to-shadow transition.
# Lower values produce a broader, softer transition.
# Higher values produce stronger separation.
# Suggested range: 0.35 to 1.20
FIELD_CONTRAST = 0.55

# Reshapes the smooth signed field without creating a hard boundary.
# 1.0 keeps the field unchanged.
# Above 1.0 creates a wider neutral middle.
# Below 1.0 spreads stronger effects over more of the image.
# Suggested range: 0.8 to 1.4
REGION_POWER = 0.80

# Weight of the image's existing broad luminance in the illumination map.
# 0.0 uses only the geometric pattern.
# Higher values make the effect follow existing broad light/dark regions.
# Suggested range: 0.0 to 0.35
IMAGE_MAP_WEIGHT = 0.35

# Mild warm shift in bright areas and cool shift in shadows.
# Set to 0.0 for no color-temperature shift.
# Suggested range: 0.0 to 0.08
TEMPERATURE_SHIFT = 0.06

# Adds a soft white veil only in the strongest illuminated regions.
# Suggested range: 0.0 to 0.08
GLARE_STRENGTH = 0.08

# Contact-sheet appearance.
CONTACT_CELL_WIDTH = 300
CONTACT_CELL_HEIGHT = 300

# =====================================================================
# END EDITABLE SETTINGS
# =====================================================================


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


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


def force_odd(value: float, minimum: int = 3) -> int:
    number = max(minimum, int(round(value)))
    return number if number % 2 == 1 else number + 1


def spatial_field(
    height: int,
    width: int,
    pattern: str,
) -> np.ndarray:
    """
    Create a broad signed field in approximately [-1, 1].

    +1 represents the strongest illuminated region.
    -1 represents the strongest shaded region.
    """
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    if pattern == "diagonal":
        field = 0.5 * (xx + yy)

    elif pattern == "horizontal":
        field = xx

    elif pattern == "vertical":
        field = yy

    elif pattern == "radial":
        radius = np.sqrt(xx**2 + yy**2) / math.sqrt(2.0)
        field = 1.0 - 2.0 * radius

    elif pattern == "vignette":
        radius = np.sqrt(xx**2 + yy**2) / math.sqrt(2.0)
        field = 2.0 * radius - 1.0

    else:
        raise ValueError(
            "PATTERN must be diagonal, horizontal, vertical, "
            "radial, or vignette."
        )

    return np.clip(field, -1.0, 1.0).astype(np.float32)


def image_luminance_field(image: np.ndarray) -> np.ndarray:
    """
    Estimate broad existing illumination while suppressing leaf texture,
    veins, lesions, and small pixel-level detail.
    """
    image = ensure_bgr(image)
    height, width = image.shape[:2]
    minimum_dimension = min(height, width)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    luminance = lab[:, :, 0].astype(np.float32) / 255.0

    sigma = float(
        np.clip(
            minimum_dimension * 0.08,
            8.0,
            45.0,
        )
    )
    kernel_size = force_odd(6.0 * sigma + 1.0)

    smooth = cv2.GaussianBlur(
        luminance,
        (kernel_size, kernel_size),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT101,
    )

    low = float(np.percentile(smooth, 5))
    high = float(np.percentile(smooth, 95))

    if high - low < 1e-6:
        return np.zeros_like(smooth, dtype=np.float32)

    normalized = np.clip(
        (smooth - low) / (high - low),
        0.0,
        1.0,
    )

    return (2.0 * normalized - 1.0).astype(np.float32)


def create_illumination_field(
    image: np.ndarray,
    pattern: str,
    image_map_weight: float,
    field_contrast: float,
    region_power: float,
) -> np.ndarray:
    """
    Create one smooth signed illumination field covering the full image.

    The result remains continuous; no threshold divides light and shadow.
    """
    image = ensure_bgr(image)
    height, width = image.shape[:2]

    image_map_weight = clamp(image_map_weight, 0.0, 1.0)

    geometric = spatial_field(height, width, pattern)
    existing = image_luminance_field(image)

    field = (
        (1.0 - image_map_weight) * geometric
        + image_map_weight * existing
    ).astype(np.float32)

    sigma = float(
        np.clip(
            min(height, width) * 0.025,
            3.0,
            18.0,
        )
    )
    kernel_size = force_odd(6.0 * sigma + 1.0)

    field = cv2.GaussianBlur(
        field,
        (kernel_size, kernel_size),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT101,
    )

    maximum = float(np.max(np.abs(field)))
    if maximum > 1e-6:
        field = field / maximum

    field = np.clip(field, -1.0, 1.0)

    if field_contrast <= 0:
        raise ValueError("FIELD_CONTRAST must be positive.")

    # Smoothly reshape the field while preserving endpoints and continuity.
    denominator = math.tanh(field_contrast)
    smooth_field = (
        np.tanh(field_contrast * field)
        / denominator
    ).astype(np.float32)

    if region_power <= 0:
        raise ValueError("REGION_POWER must be positive.")

    shaped_field = (
        np.sign(smooth_field)
        * np.power(
            np.abs(smooth_field),
            region_power,
        )
    ).astype(np.float32)

    return np.clip(
        shaped_field,
        -1.0,
        1.0,
    )


def apply_realistic_lighting(
    image: np.ndarray,
    severity: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Apply a continuous uneven-exposure corruption.

    Normalized severity:
        n = (severity / 100)^SEVERITY_GAMMA

    Smooth light/shadow blend:
        blend = (field + 1) / 2

    Exposure in stops:
        exposure =
            n * [
                -SHADOW_STOPS
                + (HIGHLIGHT_STOPS + SHADOW_STOPS) * blend
            ]

    Pixel transform:
        output = input * 2^exposure

    HIGHLIGHT_STOPS and SHADOW_STOPS directly control the saved pixels.
    """
    image = ensure_bgr(image)
    severity = clamp(float(severity), 0.0, 100.0)

    if severity <= 0:
        return image.copy(), {
            "normalized_strength": 0.0,
            "highlight_stops": 0.0,
            "shadow_stops": 0.0,
        }

    if SEVERITY_GAMMA <= 0:
        raise ValueError("SEVERITY_GAMMA must be positive.")

    if HIGHLIGHT_STOPS < 0 or SHADOW_STOPS < 0:
        raise ValueError(
            "HIGHLIGHT_STOPS and SHADOW_STOPS must be nonnegative."
        )

    normalized = (
        severity / 100.0
    ) ** SEVERITY_GAMMA

    field = create_illumination_field(
        image=image,
        pattern=PATTERN,
        image_map_weight=IMAGE_MAP_WEIGHT,
        field_contrast=FIELD_CONTRAST,
        region_power=REGION_POWER,
    )

    # Continuous 0-to-1 blend across the entire image.
    light_shadow_blend = (
        field + 1.0
    ) / 2.0

    exposure_map = normalized * (
        -SHADOW_STOPS
        + (
            HIGHLIGHT_STOPS
            + SHADOW_STOPS
        )
        * light_shadow_blend
    )

    # Work directly in normalized OpenCV BGR space.
    image_float = (
        image.astype(np.float32)
        / 255.0
    )

    # Convert photographic stops into actual brightness multipliers.
    exposure_multiplier = np.power(
        2.0,
        exposure_map,
    ).astype(np.float32)[:, :, None]

    transformed = (
        image_float
        * exposure_multiplier
    )

    # Soft continuous color-temperature weights.
    light_weight = light_shadow_blend
    shadow_weight = 1.0 - light_shadow_blend

    warm = (
        TEMPERATURE_SHIFT
        * normalized
        * light_weight
    )

    cool = (
        TEMPERATURE_SHIFT
        * normalized
        * shadow_weight
    )

    # OpenCV channel order is BGR.
    transformed[:, :, 2] *= (
        1.0
        + 0.30 * warm
        - 0.08 * cool
    )

    transformed[:, :, 1] *= (
        1.0
        + 0.08 * warm
    )

    transformed[:, :, 0] *= (
        1.0
        - 0.15 * warm
        + 0.20 * cool
    )

    # Glare is limited to the more illuminated side.
    glare_weight = np.clip(
        field,
        0.0,
        1.0,
    )

    glare = (
        GLARE_STRENGTH
        * (normalized**2.0)
        * (glare_weight**2.0)
    )[:, :, None]

    transformed = (
        transformed
        + glare * (1.0 - transformed)
    )

    output_bgr = np.clip(
        transformed * 255.0,
        0.0,
        255.0,
    ).astype(np.uint8)

    return output_bgr, {
        "normalized_strength": normalized,
        "highlight_stops": (
            normalized * HIGHLIGHT_STOPS
        ),
        "shadow_stops": (
            normalized * SHADOW_STOPS
        ),
    }


def resize_for_cell(
    image: np.ndarray,
    available_width: int,
    available_height: int,
) -> np.ndarray:
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
        else cv2.INTER_LINEAR
    )

    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=interpolation,
    )


def create_contact_sheet(
    labeled_images: list[tuple[str, np.ndarray]],
) -> np.ndarray:
    columns = len(labeled_images)

    sheet = np.full(
        (
            CONTACT_CELL_HEIGHT,
            columns * CONTACT_CELL_WIDTH,
            3,
        ),
        245,
        dtype=np.uint8,
    )

    for index, (label, image) in enumerate(labeled_images):
        preview = resize_for_cell(
            ensure_bgr(image),
            available_width=CONTACT_CELL_WIDTH - 20,
            available_height=CONTACT_CELL_HEIGHT - 55,
        )

        preview_height, preview_width = preview.shape[:2]
        x_start = index * CONTACT_CELL_WIDTH

        x_offset = (
            x_start
            + (CONTACT_CELL_WIDTH - preview_width) // 2
        )

        y_offset = (
            42
            + (
                CONTACT_CELL_HEIGHT
                - 42
                - preview_height
            ) // 2
        )

        sheet[
            y_offset:y_offset + preview_height,
            x_offset:x_offset + preview_width,
        ] = preview

        text_size, _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            1,
        )

        text_width = text_size[0]

        cv2.putText(
            sheet,
            label,
            (
                x_start
                + (CONTACT_CELL_WIDTH - text_width) // 2,
                27,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )

    return sheet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate normalized lighting previews using the constants "
            "in the EDITABLE SETTINGS section."
        )
    )

    parser.add_argument(
        "image",
        type=Path,
        help="Path to one clean source image.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("realistic_lighting_preview"),
        help="Output directory.",
    )

    return parser.parse_args()


def print_settings() -> None:
    print("\nACTIVE EDITABLE SETTINGS")
    print("=" * 55)
    print(f"PREVIEW_SEVERITIES:  {PREVIEW_SEVERITIES}")
    print(f"PATTERN:             {PATTERN}")
    print(f"SEVERITY_GAMMA:      {SEVERITY_GAMMA}")
    print(f"HIGHLIGHT_STOPS:     {HIGHLIGHT_STOPS}")
    print(f"SHADOW_STOPS:        {SHADOW_STOPS}")
    print(f"FIELD_CONTRAST:      {FIELD_CONTRAST}")
    print(f"REGION_POWER:        {REGION_POWER}")
    print(f"IMAGE_MAP_WEIGHT:    {IMAGE_MAP_WEIGHT}")
    print(f"TEMPERATURE_SHIFT:   {TEMPERATURE_SHIFT}")
    print(f"GLARE_STRENGTH:      {GLARE_STRENGTH}")
    print("=" * 55)
    print()


def main() -> None:
    args = parse_args()

    print("\nRUNNING SCRIPT:")
    print(Path(__file__).resolve())

    print_settings()

    image_path = args.image.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {image_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    labeled_images: list[tuple[str, np.ndarray]] = []

    for severity in PREVIEW_SEVERITIES:
        corrupted, parameters = apply_realistic_lighting(
            image=image,
            severity=severity,
        )

        output_name = (
            f"{image_path.stem}"
            f"_realistic_lighting_s{severity:03d}.png"
        )

        output_path = output_dir / output_name

        if not cv2.imwrite(
            str(output_path),
            corrupted,
        ):
            raise IOError(
                f"Failed to save: {output_path}"
            )

        if severity == 0:
            label = "Severity 0"
        else:
            label = (
                f"Severity {severity} | "
                f"+{parameters['highlight_stops']:.2f}/"
                f"-{parameters['shadow_stops']:.2f} EV"
            )

        labeled_images.append(
            (label, corrupted)
        )

        print(
            f"Saved severity {severity}: {output_path} "
            f"(n={parameters['normalized_strength']:.4f}, "
            f"highlight=+{parameters['highlight_stops']:.3f} EV, "
            f"shadow=-{parameters['shadow_stops']:.3f} EV)"
        )

    contact_sheet = create_contact_sheet(
        labeled_images
    )

    contact_sheet_path = (
        output_dir
        / (
            f"{image_path.stem}"
            f"_realistic_lighting_contact_sheet.png"
        )
    )

    if not cv2.imwrite(
        str(contact_sheet_path),
        contact_sheet,
    ):
        raise IOError(
            f"Failed to save: {contact_sheet_path}"
        )

    print(
        f"\nContact sheet: {contact_sheet_path}"
    )


if __name__ == "__main__":
    main()