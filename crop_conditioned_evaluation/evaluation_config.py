"""Configuration for crop-conditioned corruption evaluation.

Edit this file, then run one evaluator at a time:
    python evaluate_motion_blur.py
    python evaluate_resolution.py
    python evaluate_lighting.py
"""

from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path("/home/ethxn/Plant-Disease-Detection-main")

CORRUPTION_ROOT = PROJECT_ROOT / "crop_conditioned_corruptions"
MOTION_BLUR_DATA = CORRUPTION_ROOT / "motion_blur"
RESOLUTION_DATA = CORRUPTION_ROOT / "resolution"
LIGHTING_DATA = CORRUPTION_ROOT / "lighting"

RESULTS_ROOT = PROJECT_ROOT / "crop_conditioned_results"
MOTION_BLUR_RESULTS = RESULTS_ROOT / "motion_blur"
RESOLUTION_RESULTS = RESULTS_ROOT / "resolution"
LIGHTING_RESULTS = RESULTS_ROOT / "lighting"

# The user renamed "Flask Deployed App" to "Flask_Deployed_App".
MODEL_FILE = PROJECT_ROOT / "Flask_Deployed_App" / "CNN.py"
CHECKPOINT = (
    PROJECT_ROOT
    / "Flask_Deployed_App"
    / "plant_disease_model_1_latest.pt"
)
CLASS_MAP = PROJECT_ROOT / "Flask_Deployed_App" / "class_to_index.json"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_CLASS = "CNN"
NUM_CLASSES = 39
CONSTRUCTOR_STYLE = "positional"  # CNN(39)
STRICT_CHECKPOINT = True


# ---------------------------------------------------------------------------
# Preprocessing
# This matches the repository training transform:
# Resize shorter edge to 255, CenterCrop 224, ToTensor, no normalization.
# To match the Flask app instead, use RESIZE_MODE="resize".
# ---------------------------------------------------------------------------
IMAGE_SIZE = 224
RESIZE_MODE = "shorter-center-crop"  # resize or shorter-center-crop
RESIZE_SHORTER = 255
NORMALIZATION = "none"  # none, imagenet, or custom
CUSTOM_MEAN = (0.0, 0.0, 0.0)
CUSTOM_STD = (1.0, 1.0, 1.0)


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
DEVICE = "cpu"
BATCH_SIZE = 32
NUM_WORKERS = 2
PREFETCH_FACTOR = 2
USE_AMP = False

# Save a JSON copy of the exact settings used inside each results folder.
SAVE_CONFIG_SNAPSHOT = True


# ---------------------------------------------------------------------------
# Crop groups used for both analysis and crop-conditioned prediction.
# For a known Apple image, the conditioned prediction can only choose one of
# the four Apple outputs; similarly for Corn, Grape, and Tomato.
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
