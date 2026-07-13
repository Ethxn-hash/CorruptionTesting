CROP-ONLY CORRUPTION GENERATION
===============================

Purpose
-------
Generate corrupted images only for Apple, Corn, Grape, and Tomato classes.
The same deterministic clean-image manifest is reused for motion blur,
resolution degradation, and lighting variation.

Files
-----
generation_config.py          Edit all paths and settings here.
crop_corruption_common.py     Shared sampling and file utilities.
generate_motion_blur.py       Motion-blur generation.
generate_resolution.py        Resolution-degradation generation.
generate_lighting.py          Lighting-variation generation.

Setup
-----
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Configuration
-------------
Open generation_config.py and set CLEAN_DATASET and OUTPUT_ROOT.
CLEAN_DATASET must be the folder whose immediate subfolders are class names.

The configured class groups contain:
- 4 Apple classes
- 4 Corn classes
- 4 Grape classes
- 10 Tomato classes

Sampling per class is:
selected = min(available, max(MIN_PER_CLASS, round(SAMPLE_RATE * available)))

The first factor script creates MANIFEST_PATH. Later factor scripts load the
same manifest, ensuring identical clean source images across all factors.
Set RECREATE_MANIFEST=True only when intentionally choosing a new subset.

Run separately
--------------
python generate_motion_blur.py
python generate_resolution.py
python generate_lighting.py

Each output dataset keeps class folders as its immediate subfolders and writes
an index CSV inside the output root. Severity 0 is copied byte-for-byte from the
clean source image.
