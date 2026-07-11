import argparse
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def nearest_odd(value):
    n = max(1, int(round(value)))
    if n % 2 == 0:
        n += 1
    return n


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


def motion_kernel_size(severity, height, width, max_blur_fraction=0.30, gamma=2.0):
    """
    S in [0,100]
    m = min(H,W)
    Lmax = nearest_odd(max_blur_fraction * m)
    k(S) = nearest_odd(1 + (Lmax - 1) * (S/100)^gamma)
    """
    severity = clamp(severity, 0, 100)
    m = min(height, width)
    l_max = nearest_odd(max_blur_fraction * m)
    return nearest_odd(1 + (l_max - 1) * ((severity / 100.0) ** gamma))


def create_motion_blur_kernel(kernel_size, angle_degrees):
    if kernel_size <= 1:
        return np.array([[1.0]], dtype=np.float32)

    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2
    kernel[center, :] = 1.0

    rotation_matrix = cv2.getRotationMatrix2D((center, center), angle_degrees, 1.0)

    rotated_kernel = cv2.warpAffine(
        kernel,
        rotation_matrix,
        (kernel_size, kernel_size)
    )

    kernel_sum = rotated_kernel.sum()
    if kernel_sum != 0:
        rotated_kernel /= kernel_sum

    return rotated_kernel


def apply_motion_blur(image, severity, angle_degrees=0.0, max_blur_fraction=0.30, gamma=2.0):
    height, width = image.shape[:2]
    kernel_size = motion_kernel_size(
        severity,
        height,
        width,
        max_blur_fraction=max_blur_fraction,
        gamma=gamma
    )
    kernel = create_motion_blur_kernel(kernel_size, angle_degrees)
    blurred = cv2.filter2D(image, ddepth=-1, kernel=kernel)
    return blurred, kernel_size


def main():
    parser = argparse.ArgumentParser(description="Generate motion blur corruptions.")
    parser.add_argument("--input", required=True, help="Input image or folder.")
    parser.add_argument("--output", required=True, help="Output folder.")
    parser.add_argument("--severity-step", type=int, default=5, help="Default: 5")
    parser.add_argument("--severities", default=None, help="Optional list like 0,25,50,75,100")
    parser.add_argument("--angle", type=float, default=0.0, help="Blur angle in degrees.")
    parser.add_argument("--max-blur-fraction", type=float, default=0.30,
                        help="Severity 100 blur length as fraction of min(H,W).")
    parser.add_argument("--gamma", type=float, default=2.0,
                        help="Severity curve; larger keeps low severities milder.")
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
            corrupted, kernel_size = apply_motion_blur(
                image,
                severity=severity,
                angle_degrees=args.angle,
                max_blur_fraction=args.max_blur_fraction,
                gamma=args.gamma
            )

            s_text = severity_string(severity)
            output_name = (
                f"{image_path.stem}_motion_s{s_text}"
                f"_k{kernel_size}_a{int(args.angle)}{image_path.suffix}"
            )

            output_path = build_output_path(image_path, input_path, output_dir, output_name)
            cv2.imwrite(str(output_path), corrupted)
            print(f"Saved: {output_path}")
            count += 1

    print(f"\nDone. Generated {count} images.")


if __name__ == "__main__":
    main()