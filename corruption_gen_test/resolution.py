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


def apply_resolution_degradation(
    image,
    severity,
    upsample_method="linear",
):
    """
    Apply normalized resolution degradation.

    Severity 0:
        Original image.

    Severity 100:
        Reduce to exactly 1 x 1 pixel and enlarge back.
    """

    severity = float(np.clip(severity, 0, 100))

    if severity <= 0:
        return image.copy(), 1.0, image.shape[1], image.shape[0]

    height, width = image.shape[:2]

    if severity >= 100:
        reduced_width = 1
        reduced_height = 1
        scale = 0.0
    else:
        scale = float(ns.get_resolution_scale(severity))
        scale = float(np.clip(scale, 0.0, 1.0))

        reduced_width = max(
            1,
            int(round(width * scale)),
        )

        reduced_height = max(
            1,
            int(round(height * scale)),
        )

    reduced = cv2.resize(
        image,
        (reduced_width, reduced_height),
        interpolation=cv2.INTER_AREA,
    )

    interpolation_methods = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
    }

    restored = cv2.resize(
        reduced,
        (width, height),
        interpolation=interpolation_methods[upsample_method],
    )

    return restored, scale, reduced_width, reduced_height


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
    cell_width=270,
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
            0.54,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return sheet


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate normalized resolution degradation "
            "for severity levels 0 through 100."
        )
    )

    parser.add_argument(
        "image",
        help="Path to the input image.",
    )

    parser.add_argument(
        "--output_dir",
        default="resolution_test",
        help="Output directory. Default: resolution_test",
    )

    parser.add_argument(
        "--upsample_method",
        choices=["nearest", "linear", "cubic"],
        default="linear",
        help="Method used to enlarge the reduced image.",
    )

    args = parser.parse_args()

    image = read_image(args.image)
    ensure_directory(args.output_dir)

    base_name = os.path.splitext(
        os.path.basename(args.image)
    )[0]

    contact_images = []

    for severity in SEVERITIES:
        corrupted, scale, reduced_width, reduced_height = (
            apply_resolution_degradation(
                image,
                severity,
                upsample_method=args.upsample_method,
            )
        )

        output_name = (
            f"{base_name}_resolution_s{int(severity):03d}"
            f"_Sc{scale:.4f}.png"
        )

        output_path = os.path.join(
            args.output_dir,
            output_name,
        )

        if not cv2.imwrite(output_path, corrupted):
            raise IOError(f"Failed to save: {output_path}")

        parameter_text = (
            f"{reduced_width}x{reduced_height}"
        )

        contact_images.append(
            (
                f"Severity {severity} | {parameter_text}",
                corrupted,
            )
        )

        print(
            f"Saved severity {severity}: {output_path} "
            f"(scale={scale:.4f}, reduced={parameter_text})"
        )

    contact_sheet = create_contact_sheet(contact_images)

    sheet_path = os.path.join(
        args.output_dir,
        f"{base_name}_resolution_contact_sheet.png",
    )

    if not cv2.imwrite(sheet_path, contact_sheet):
        raise IOError(f"Failed to save: {sheet_path}")

    print(f"\nContact sheet: {sheet_path}")


if __name__ == "__main__":
    main()