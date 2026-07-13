"""Configuration for crop-only corruption generation.

Edit this file, then run one of:
    python generate_motion_blur.py
    python generate_resolution.py
    python generate_lighting.py

The same manifest is reused for every factor so every corruption is applied to
exactly the same clean source images.
"""

from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path("/home/ethxn/Plant-Disease-Detection-main")

# Folder whose immediate subfolders are the clean class folders.
CLEAN_DATASET = PROJECT_ROOT / "Dataset"

# All generated datasets and the shared manifest will be placed here.
OUTPUT_ROOT = PROJECT_ROOT / "crop_conditioned_corruptions"
MANIFEST_PATH = OUTPUT_ROOT / "selected_crop_images.csv"

MOTION_BLUR_OUTPUT = OUTPUT_ROOT / "motion_blur"
RESOLUTION_OUTPUT = OUTPUT_ROOT / "resolution"
LIGHTING_OUTPUT = OUTPUT_ROOT / "lighting"


# ---------------------------------------------------------------------------
# Crop groups
# Exact names must match the clean dataset folders and class map.
# ---------------------------------------------------------------------------
CROP_CLASS_GROUPS = {
    "Apple": [
        "Apple___Apple_scab",
        "Apple___Black_rot",
        "Apple___Cedar_apple_rust",
        "Apple___healthy",
    ],
    "Corn": [
        "Corn___Cercospora_leaf_spot Gray_leaf_spot",
        "Corn___Common_rust",
        "Corn___Northern_Leaf_Blight",
        "Corn___healthy",
    ],
    "Grape": [
        "Grape___Black_rot",
        "Grape___Esca_(Black_Measles)",
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
        "Grape___healthy",
    ],
    "Tomato": [
        "Tomato___Bacterial_spot",
        "Tomato___Early_blight",
        "Tomato___Late_blight",
        "Tomato___Leaf_Mold",
        "Tomato___Septoria_leaf_spot",
        "Tomato___Spider_mites Two-spotted_spider_mite",
        "Tomato___Target_Spot",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "Tomato___Tomato_mosaic_virus",
        "Tomato___healthy",
    ],
}


# ---------------------------------------------------------------------------
# Shared sampling settings
# n_class = min(available, max(MIN_PER_CLASS, round(SAMPLE_RATE * available)))
# ---------------------------------------------------------------------------
SEVERITIES = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
MIN_PER_CLASS = 100
SAMPLE_RATE = 0.086
RANDOM_SEED = 2026

# Set True only when you intentionally want a new random selection.
RECREATE_MANIFEST = False

# Existing generated files are skipped unless this is True.
OVERWRITE = False
PROGRESS_EVERY = 100
JPEG_QUALITY = 95


# ---------------------------------------------------------------------------
# Motion blur
# k(S) = nearest_odd(1 + (Lmax - 1) * (S/100)^gamma)
# Lmax = nearest_odd(MAX_BLUR_FRACTION * min(height, width))
# ---------------------------------------------------------------------------
MOTION_BLUR_ANGLE_DEGREES = 0.0
MOTION_BLUR_MAX_FRACTION = 0.30
MOTION_BLUR_GAMMA = 2.0


# ---------------------------------------------------------------------------
# Resolution degradation
# scale(S) = 1 - (1 - MIN_SCALE) * (S/100)
# ---------------------------------------------------------------------------
RESOLUTION_MIN_SCALE = 0.03
RESOLUTION_UPSAMPLE_METHOD = "linear"  # nearest, linear, or cubic


# ---------------------------------------------------------------------------
# Lighting variation
# ---------------------------------------------------------------------------
LIGHTING_MIN_MULTIPLIER = 0.0
LIGHTING_MAX_MULTIPLIER = 6.0
LIGHTING_SHARPNESS = 12.0
LIGHTING_PATTERN = "diagonal"  # diagonal, horizontal, vertical, radial, vignette
