import argparse
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def clamp(value, low, high):
    return max(low, min(high, value))


def parse_severities(severity_text, severity_step):
    if severity_text:
        severities = [float(x.strip()) for x in severity_text.split(",") if x.strip()]
    else:
        severities = list(range(0, 101, severity_step))
        if severities[-1] != 100:
            severities.append(100)

    for s in severities:
        if s < 0 or s > 100:
            raise ValueError(f"Severity must be in [0, 100], got {s}")

    return severities


def severity_string(severity):
    if float(severity).is_integer():
        return str(int(severity))
    return str(severity).replace(".", "p")


def collect_images(input_path):
    input_path = Path(input_path)

    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image file: {input_path}")
        return [input_path]

    if input_path.is_dir():
        return sorted(
            p for p in input_path.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def build_output_path(image_path, input_path, output_dir, output_name):
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    if input_path.is_dir():
        relative_parent = image_path.parent.relative_to(input_path)
        final_dir = output_dir / relative_parent
    else:
        final_dir = output_dir

    final_dir.mkdir(parents=True, exist_ok=True)
    return final_dir / output_name


def lighting_pattern(height, width, pattern="diagonal"):
    """
    Creates P(x,y), a spatial pattern in [0,1].

    diagonal:
        dark-to-bright from top-left to bottom-right

    horizontal:
        dark-to-bright from left to right

    vertical:
        dark-to-bright from top to bottom

    radial:
        brighter center, darker edges

    vignette:
        darker center, brighter edges
    """
    x = np.linspace(0, 1, width, dtype=np.float32)
    y = np.linspace(0, 1, height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    if pattern == "diagonal":
        p = 0.5 * (xx + yy)

    elif pattern == "horizontal":
        p = xx

    elif pattern == "vertical":
        p = yy

    elif pattern == "radial":
        distance = np.sqrt((xx - 0.5) ** 2 + (yy - 0.5) ** 2)
        max_distance = np.sqrt(0.5 ** 2 + 0.5 ** 2)
        p = 1.0 - distance / max_distance

    elif pattern == "vignette":
        distance = np.sqrt((xx - 0.5) ** 2 + (yy - 0.5) ** 2)
        max_distance = np.sqrt(0.5 ** 2 + 0.5 ** 2)
        p = distance / max_distance

    else:
        raise ValueError(
            "pattern must be diagonal, horizontal, vertical, radial, or vignette"
        )

    return np.clip(p, 0, 1).astype(np.float32)


def apply_lighting_variation(
    image,
    severity,
    min_multiplier=0.00,
    max_multiplier=6.00,
    sharpness=12.0,
    pattern="diagonal"
):
    """
    Severe normalized lighting corruption.

    Severity:
        S in [0, 100]

    Linear severity strength:
        alpha = S / 100

    Spatial pattern:
        P(x,y) in [0,1]

    Harsh lighting split:
        B(x,y) = 1 / (1 + exp(-sharpness * (P(x,y) - 0.5)))

    Target multiplier:
        T(x,y) = M_min + (M_max - M_min) * B(x,y)

    Final multiplier:
        M_S(x,y) = (1 - alpha) + alpha * T(x,y)

    Corrupted image:
        I_S(x,y) = clip(I(x,y) * M_S(x,y), 0, 255)

    At S = 0:
        alpha = 0, so image is unchanged.

    At S = 100:
        alpha = 1, so the full severe lighting corruption is applied.
    """

    height, width = image.shape[:2]

    severity = clamp(severity, 0, 100)
    alpha = severity / 100.0

    p = lighting_pattern(height, width, pattern)

    # Sigmoid creates a harsher dark/bright split than a smooth gradient.
    b = 1.0 / (1.0 + np.exp(-sharpness * (p - 0.5)))

    target_multiplier = min_multiplier + (max_multiplier - min_multiplier) * b

    multiplier = (1.0 - alpha) + alpha * target_multiplier

    if image.ndim == 3:
        multiplier = multiplier[:, :, None]

    corrupted = image.astype(np.float32) * multiplier
    corrupted = np.clip(corrupted, 0, 255).astype(np.uint8)

    return corrupted


def main():
    parser = argparse.ArgumentParser(
        description="Generate severe normalized lighting variation corruptions."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input image or folder."
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output folder."
    )

    parser.add_argument(
        "--severity-step",
        type=int,
        default=5,
        help="Generate severity levels from 0 to 100 by this step. Default is 5."
    )

    parser.add_argument(
        "--severities",
        default=None,
        help="Optional list like 0,25,50,75,100."
    )

    parser.add_argument(
        "--min-multiplier",
        type=float,
        default=0.00,
        help="Darkest lighting multiplier at severity 100. Default is 0.00."
    )

    parser.add_argument(
        "--max-multiplier",
        type=float,
        default=6.00,
        help="Brightest lighting multiplier at severity 100. Default is 6.00."
    )

    parser.add_argument(
        "--sharpness",
        type=float,
        default=12.0,
        help="Higher value creates a sharper dark/bright split. Default is 12.0."
    )

    parser.add_argument(
        "--pattern",
        default="diagonal",
        choices=["diagonal", "horizontal", "vertical", "radial", "vignette"],
        help="Lighting pattern. Default is diagonal."
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    severities = parse_severities(args.severities, args.severity_step)
    image_paths = collect_images(input_path)

    print(f"Input path: {input_path}")
    print(f"Output folder: {output_dir}")
    print(f"Found {len(image_paths)} image(s).")

    if len(image_paths) == 0:
        print("No images found. Check your input path and file extensions.")
        return

    count = 0

    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if image is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        for severity in severities:
            corrupted = apply_lighting_variation(
                image=image,
                severity=severity,
                min_multiplier=args.min_multiplier,
                max_multiplier=args.max_multiplier,
                sharpness=args.sharpness,
                pattern=args.pattern
            )

            s_text = severity_string(severity)

            output_name = (
                f"{image_path.stem}"
                f"_lighting_s{s_text}"
                f"_min{str(args.min_multiplier).replace('.', 'p')}"
                f"_max{str(args.max_multiplier).replace('.', 'p')}"
                f"_sh{str(args.sharpness).replace('.', 'p')}"
                f"{image_path.suffix}"
            )

            output_path = build_output_path(
                image_path=image_path,
                input_path=input_path,
                output_dir=output_dir,
                output_name=output_name
            )

            cv2.imwrite(str(output_path), corrupted)
            print(f"Saved: {output_path}")
            count += 1

    print(f"\nDone. Generated {count} images.")


if __name__ == "__main__":
    main()