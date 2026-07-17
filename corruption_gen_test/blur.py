import argparse
import math
import os

import cv2
import numpy as np

import normalized_severity as ns


SEVERITIES = getattr(ns, "SEVERITIES", list(range(0, 101, 10)))


def read_image(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {path}\n"
            "Check the path, filename, and extension."
        )

    return image


def ensure_directory(path):
    os.makedirs(path, exist_ok=True)


def force_odd(value):
    value = max(1, int(round(value)))

    if value % 2 == 0:
        value += 1

    return value


def make_motion_blur_kernel(length, angle_degrees):
    """
    Create a normalized linear motion-blur kernel.
    """

    length = force_odd(length)

    if length <= 1:
        return np.array([[1.0]], dtype=np.float32)

    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0

    center = (
        (length - 1) / 2.0,
        (length - 1) / 2.0,
    )

    rotation_matrix = cv2.getRotationMatrix2D(
        center,
        angle_degrees,
        1.0,
    )

    kernel = cv2.warpAffine(
        kernel,
        rotation_matrix,
        (length, length),
        flags=cv2.INTER_LINEAR,
    )

    kernel_sum = float(kernel.sum())

    if kernel_sum <= 0:
        return np.array([[1.0]], dtype=np.float32)

    return kernel / kernel_sum


def apply_motion_blur(image, severity, angle_degrees=15.0):
    """
    Apply normalized motion blur.

    Severity 0:
        Original image.

    Severity 100:
        Complete information collapse to the image's mean color.
    """

    severity = float(np.clip(severity, 0, 100))

    if severity <= 0:
        return image.copy(), 1

    if severity >= 100:
        mean_color = np.mean(
            image.astype(np.float32),
            axis=(0, 1),
            keepdims=True,
        )

        collapsed = np.broadcast_to(
            mean_color,
            image.shape,
        ).copy()

        return np.clip(collapsed, 0, 255).astype(np.uint8), 0

    blur_length = ns.get_blur_length(severity)

    if blur_length is None:
        raise ValueError(
            "get_blur_length() returned None below severity 100."
        )

    blur_length = force_odd(blur_length)

    # Avoid kernels larger than the image's smallest dimension.
    maximum_length = max(1, min(image.shape[:2]))

    if maximum_length % 2 == 0:
        maximum_length -= 1

    maximum_length = max(1, maximum_length)
    blur_length = min(blur_length, maximum_length)

    kernel = make_motion_blur_kernel(
        blur_length,
        angle_degrees,
    )

    corrupted = cv2.filter2D(
        image,
        ddepth=-1,
        kernel=kernel,
        borderType=cv2.BORDER_REFLECT101,
    )

    return corrupted, blur_length


def resize_for_cell(image, available_width, available_height):
    height, width = image.shape[:2]

    scale = min(
        available_width / width,
        available_height / height,
    )

    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))

    interpolation = (
        cv2.INTER_AREA if scale < 1.0
        else cv2.INTER_NEAREST
    )

    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=interpolation,
    )


def create_contact_sheet(
    labeled_images,
    columns=3,
    cell_width=260,
    cell_height=250,
):
    rows = math.ceil(len(labeled_images) / columns)

    sheet = np.full(
        (rows * cell_height, columns * cell_width, 3),
        255,
        dtype=np.uint8,
    )

    for index, (label, image) in enumerate(labeled_images):
        row = index // columns
        column = index % columns

        x_start = column * cell_width
        y_start = row * cell_height

        preview = resize_for_cell(
            image,
            available_width=cell_width - 20,
            available_height=cell_height - 50,
        )

        preview_height, preview_width = preview.shape[:2]

        x_offset = x_start + (cell_width - preview_width) // 2
        y_offset = y_start + 38 + (
            cell_height - 38 - preview_height
        ) // 2

        sheet[
            y_offset:y_offset + preview_height,
            x_offset:x_offset + preview_width,
        ] = preview

        cv2.putText(
            sheet,
            label,
            (x_start + 10, y_start + 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return sheet


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate normalized motion-blur corruption "
            "for severity levels 0 through 100."
        )
    )

    parser.add_argument(
        "image",
        help="Path to the input image.",
    )

    parser.add_argument(
        "--output_dir",
        default="blur_test",
        help="Output directory. Default: blur_test",
    )

    parser.add_argument(
        "--angle",
        type=float,
        default=15.0,
        help="Motion-blur angle in degrees. Default: 15",
    )

    args = parser.parse_args()

    image = read_image(args.image)
    ensure_directory(args.output_dir)

    base_name = os.path.splitext(
        os.path.basename(args.image)
    )[0]

    contact_images = []

    for severity in SEVERITIES:
        corrupted, blur_length = apply_motion_blur(
            image,
            severity,
            angle_degrees=args.angle,
        )

        output_name = (
            f"{base_name}_blur_s{int(severity):03d}.png"
        )

        output_path = os.path.join(
            args.output_dir,
            output_name,
        )

        if not cv2.imwrite(output_path, corrupted):
            raise IOError(f"Failed to save: {output_path}")

        parameter_text = (
            "collapse" if severity >= 100
            else f"k={blur_length}"
        )

        contact_images.append(
            (
                f"Severity {severity} | {parameter_text}",
                corrupted,
            )
        )

        print(
            f"Saved severity {severity}: "
            f"{output_path} ({parameter_text})"
        )

    contact_sheet = create_contact_sheet(contact_images)

    sheet_path = os.path.join(
        args.output_dir,
        f"{base_name}_blur_contact_sheet.png",
    )

    if not cv2.imwrite(sheet_path, contact_sheet):
        raise IOError(f"Failed to save: {sheet_path}")

    print(f"\nContact sheet: {sheet_path}")


if __name__ == "__main__":
    main()