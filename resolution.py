import argparse
from pathlib import Path

import cv2


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


def float_to_filename(value, decimals=4):
    return f"{value:.{decimals}f}".replace(".", "p")


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


def resolution_scale_factor(severity, min_scale=0.03):
    severity = clamp(severity, 0, 100)
    scale = 1.0 - (1.0 - min_scale) * (severity / 100.0)
    return clamp(scale, min_scale, 1.0)


def apply_resolution_degradation(image, severity, min_scale=0.03, gamma=2.0,
                                 upsample_method="linear"):
    height, width = image.shape[:2]

    scale = resolution_scale_factor(severity, min_scale=min_scale)

    small_width = max(1, int(round(width * scale)))
    small_height = max(1, int(round(height * scale)))

    small = cv2.resize(
        image,
        (small_width, small_height),
        interpolation=cv2.INTER_AREA
    )

    if upsample_method == "nearest":
        interpolation = cv2.INTER_NEAREST
    elif upsample_method == "linear":
        interpolation = cv2.INTER_LINEAR
    elif upsample_method == "cubic":
        interpolation = cv2.INTER_CUBIC
    else:
        raise ValueError("upsample_method must be nearest, linear, or cubic")

    degraded = cv2.resize(
        small,
        (width, height),
        interpolation=interpolation
    )

    return degraded, scale


def main():
    parser = argparse.ArgumentParser(description="Generate resolution degradation corruptions.")
    parser.add_argument("--input", required=True, help="Input image or folder.")
    parser.add_argument("--output", required=True, help="Output folder.")
    parser.add_argument("--severity-step", type=int, default=5, help="Default: 5")
    parser.add_argument("--severities", default=None, help="Optional list like 0,25,50,75,100")
    parser.add_argument("--min-scale", type=float, default=0.03,
                        help="Downsample factor at severity 100.")
    parser.add_argument("--upsample-method", default="linear",
                        choices=["nearest", "linear", "cubic"],
                        help="Method used to resize back to original size.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    severities = parse_severities(args.severities, args.severity_step)
    image_paths = collect_images(input_path)

    count = 0

    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        for severity in severities:
            corrupted, scale = apply_resolution_degradation(
                image,
                severity=severity,
                min_scale=args.min_scale,   
                upsample_method=args.upsample_method
            )

            s_text = severity_string(severity)
            output_name = (
                f"{image_path.stem}_resolution_s{s_text}"
                f"_Sc{float_to_filename(scale, 4)}{image_path.suffix}"
            )

            output_path = build_output_path(image_path, input_path, output_dir, output_name)
            cv2.imwrite(str(output_path), corrupted)
            print(f"Saved: {output_path}")
            count += 1

    print(f"\nDone. Generated {count} images.")


if __name__ == "__main__":
    main()