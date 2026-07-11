from pathlib import Path
from types import SimpleNamespace

# EDIT THESE PATHS ONCE
PROJECT_ROOT = Path("/home/ethxn/Plant-Disease-Detection-main")

MOTION_BLUR_DATA = PROJECT_ROOT / "PlantVillage_motion_blur"
RESOLUTION_DATA = PROJECT_ROOT / "PlantVillage_resolution"
LIGHTING_DATA = PROJECT_ROOT / "PlantVillage_lighting"
RESULTS_ROOT = PROJECT_ROOT / "corruption_evaluation_results"

MODEL_FILE = PROJECT_ROOT / "CNN.py"
CHECKPOINT = PROJECT_ROOT / "model.pth"
CLASS_MAP = PROJECT_ROOT / "class_to_index.json"

# MODEL / PREPROCESSING SETTINGS
MODEL_CLASS = "CNN"
NUM_CLASSES = None
CONSTRUCTOR_STYLE = "auto"
NON_STRICT_CHECKPOINT = False

IMAGE_SIZE = 256
RESIZE_MODE = "resize"
RESIZE_SHORTER = 256
NORMALIZATION = "none"
CUSTOM_MEAN = "0.485,0.456,0.406"
CUSTOM_STD = "0.229,0.224,0.225"

# PERFORMANCE SETTINGS
DEVICE = "auto"       # auto, cpu, cuda, cuda:0, or mps
BATCH_SIZE = 64
NUM_WORKERS = 4
PREFETCH_FACTOR = 2
USE_AMP = True        # used only on CUDA

# ENABLE/DISABLE FACTORS FOR run_all_evaluations.py
RUN_MOTION_BLUR = True
RUN_RESOLUTION = True
RUN_LIGHTING = True

FACTOR_PATHS = {
    "motion_blur": {
        "data": MOTION_BLUR_DATA,
        "output": RESULTS_ROOT / "motion_blur",
    },
    "resolution": {
        "data": RESOLUTION_DATA,
        "output": RESULTS_ROOT / "resolution",
    },
    "lighting": {
        "data": LIGHTING_DATA,
        "output": RESULTS_ROOT / "lighting",
    },
}

def make_args(factor: str) -> SimpleNamespace:
    paths = FACTOR_PATHS[factor]
    return SimpleNamespace(
        data=str(paths["data"]),
        output=str(paths["output"]),
        model_file=str(MODEL_FILE),
        model_class=MODEL_CLASS,
        checkpoint=str(CHECKPOINT),
        class_map=str(CLASS_MAP) if CLASS_MAP is not None else None,
        num_classes=NUM_CLASSES,
        constructor_style=CONSTRUCTOR_STYLE,
        non_strict_checkpoint=NON_STRICT_CHECKPOINT,
        image_size=IMAGE_SIZE,
        resize_mode=RESIZE_MODE,
        resize_shorter=RESIZE_SHORTER,
        normalization=NORMALIZATION,
        mean=CUSTOM_MEAN,
        std=CUSTOM_STD,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        device=DEVICE,
        amp=USE_AMP,
    )
