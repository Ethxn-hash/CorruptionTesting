"""
Shared normalized-severity lookup tables for corruption generation.

Required by:
    blur.py
    resolution.py
    lighting.py

Severity levels:
    0, 10, 20, ..., 100

Important:
These lookup tables provide a consistent severity interface and a visually
progressive starting scale. They are not yet experimentally calibrated using
LPIPS, SSIM, or another perceptual metric across a representative image set.
"""

from bisect import bisect_right
from typing import Mapping


SEVERITIES = list(range(0, 101, 10))


# ---------------------------------------------------------------------
# MOTION BLUR
# ---------------------------------------------------------------------
# Values represent motion-blur kernel lengths in pixels.
#
# The progression is nonlinear because kernel length does not produce a
# visually linear change when increased linearly.
#
# blur.py handles severity 100 separately by collapsing the image, so the
# final table value is mainly included for completeness.
BLUR_LENGTH_TABLE = {
    0: 1,
    10: 3,
    20: 7,
    30: 13,
    40: 23,
    50: 39,
    60: 63,
    70: 95,
    80: 139,
    90: 191,
    100: 255,
}


# ---------------------------------------------------------------------
# RESOLUTION DEGRADATION
# ---------------------------------------------------------------------
# Values represent the fraction of the original width and height retained
# during downsampling.
#
# Example:
#     scale = 0.50
# means that an image is reduced to 50% of its original width and height.
#
# resolution.py handles severity 100 separately as a 1 x 1 reconstruction.
RESOLUTION_SCALE_TABLE = {
    0: 1.000,
    10: 0.780,
    20: 0.580,
    30: 0.420,
    40: 0.300,
    50: 0.210,
    60: 0.140,
    70: 0.090,
    80: 0.050,
    90: 0.020,
    100: 0.000,
}


# ---------------------------------------------------------------------
# LIGHTING: BRIGHT/DARK EXPANSION
# ---------------------------------------------------------------------
# Controls how strongly existing bright regions are pushed toward white
# and existing dark regions are pushed toward black.
#
# 0.0 = no change
# 1.0 = maximum push toward the exposure endpoints
LIGHTING_PUSH_TABLE = {
    0: 0.00,
    10: 0.10,
    20: 0.20,
    30: 0.30,
    40: 0.40,
    50: 0.50,
    60: 0.60,
    70: 0.70,
    80: 0.80,
    90: 0.90,
    100: 1.00,
}


# ---------------------------------------------------------------------
# LIGHTING: BLACK/WHITE TRANSITION
# ---------------------------------------------------------------------
# Controls the gradual transition from harsh illumination toward the final
# black-and-white destroyed endpoint.
#
# The transition begins gently before the final levels so that severity 100
# is not an abrupt visual jump from severity 90.
LIGHTING_BW_BLEND_TABLE = {
    0: 0.00,
    10: 0.00,
    20: 0.00,
    30: 0.02,
    40: 0.06,
    50: 0.12,
    60: 0.22,
    70: 0.36,
    80: 0.54,
    90: 0.76,
    100: 1.00,
}


def clamp_severity(severity: float) -> float:
    """Restrict severity to the valid range of 0 through 100."""

    try:
        value = float(severity)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Severity must be numeric, received: {severity!r}"
        ) from error

    return max(0.0, min(100.0, value))


def validate_table(
    table: Mapping[int, float],
    table_name: str,
) -> None:
    """
    Verify that a severity table contains the required endpoints and
    monotonically increasing severity keys.
    """

    if not table:
        raise ValueError(f"{table_name} cannot be empty.")

    keys = sorted(table)

    if keys[0] != 0:
        raise ValueError(
            f"{table_name} must include severity 0."
        )

    if keys[-1] != 100:
        raise ValueError(
            f"{table_name} must include severity 100."
        )

    if len(keys) != len(set(keys)):
        raise ValueError(
            f"{table_name} contains duplicate severity keys."
        )

    for key in keys:
        if key < 0 or key > 100:
            raise ValueError(
                f"{table_name} has invalid severity key: {key}"
            )


def interpolate_table(
    table: Mapping[int, float],
    severity: float,
) -> float:
    """
    Linearly interpolate between neighboring entries in a lookup table.

    This allows the programs to support both the standard 10-point levels
    and intermediate severities such as 35 or 67.
    """

    severity = clamp_severity(severity)
    keys = sorted(table)

    if severity <= keys[0]:
        return float(table[keys[0]])

    if severity >= keys[-1]:
        return float(table[keys[-1]])

    exact_key = int(severity)

    if severity.is_integer() and exact_key in table:
        return float(table[exact_key])

    upper_index = bisect_right(keys, severity)
    lower_key = keys[upper_index - 1]
    upper_key = keys[upper_index]

    lower_value = float(table[lower_key])
    upper_value = float(table[upper_key])

    fraction = (
        (severity - lower_key)
        / (upper_key - lower_key)
    )

    return (
        lower_value
        + fraction * (upper_value - lower_value)
    )


def make_odd(value: float) -> int:
    """Round a value to the nearest positive odd integer."""

    result = max(1, int(round(value)))

    if result % 2 == 0:
        result += 1

    return result


def get_blur_length(severity: float) -> int:
    """
    Return the normalized motion-blur kernel length.

    The returned value is always a positive odd integer.
    """

    value = interpolate_table(
        BLUR_LENGTH_TABLE,
        severity,
    )

    return make_odd(value)


def get_resolution_scale(severity: float) -> float:
    """
    Return the normalized downsampling scale.

    Returns a value between 0.0 and 1.0.
    """

    value = interpolate_table(
        RESOLUTION_SCALE_TABLE,
        severity,
    )

    return max(0.0, min(1.0, value))


def get_lighting_push(severity: float) -> float:
    """
    Return the bright/dark illumination expansion strength.

    Returns a value between 0.0 and 1.0.
    """

    value = interpolate_table(
        LIGHTING_PUSH_TABLE,
        severity,
    )

    return max(0.0, min(1.0, value))


def get_lighting_bw_blend(severity: float) -> float:
    """
    Return the blend toward the black-and-white lighting endpoint.

    Returns a value between 0.0 and 1.0.
    """

    value = interpolate_table(
        LIGHTING_BW_BLEND_TABLE,
        severity,
    )

    return max(0.0, min(1.0, value))


def print_severity_table() -> None:
    """Print all shared parameter values for inspection."""

    header = (
        f"{'Severity':>8} | "
        f"{'Blur':>7} | "
        f"{'Resolution':>10} | "
        f"{'Light push':>10} | "
        f"{'BW blend':>8}"
    )

    print(header)
    print("-" * len(header))

    for severity in SEVERITIES:
        print(
            f"{severity:>8} | "
            f"{get_blur_length(severity):>7} | "
            f"{get_resolution_scale(severity):>10.3f} | "
            f"{get_lighting_push(severity):>10.2f} | "
            f"{get_lighting_bw_blend(severity):>8.2f}"
        )


validate_table(
    BLUR_LENGTH_TABLE,
    "BLUR_LENGTH_TABLE",
)

validate_table(
    RESOLUTION_SCALE_TABLE,
    "RESOLUTION_SCALE_TABLE",
)

validate_table(
    LIGHTING_PUSH_TABLE,
    "LIGHTING_PUSH_TABLE",
)

validate_table(
    LIGHTING_BW_BLEND_TABLE,
    "LIGHTING_BW_BLEND_TABLE",
)


if __name__ == "__main__":
    print_severity_table()