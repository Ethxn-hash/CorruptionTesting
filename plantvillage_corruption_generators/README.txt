PLANTVILLAGE CORRUPTION GENERATORS
==================================

FILES
-----
corruption_common.py
generate_motion_blur_dataset.py
generate_resolution_dataset.py
generate_lighting_dataset.py

DEPENDENCIES
------------
Python 3.8+
pip install numpy opencv-python

INPUT STRUCTURE
---------------
Point --input at the folder whose immediate subfolders are the classes:

PlantVillage_clean/
    Apple___Apple_scab/
        image1.JPG
    Apple___Black_rot/
        image2.JPG
    ...

SAMPLING STRATEGY
-----------------
The default hybrid allocation is:

selected_for_class =
    min(available, max(100, round(0.086 * available)))

For the standard 38-class, 54,305-image PlantVillage distribution, this selects
5,450 unique source images. Each selected image is generated at 11 severity
levels: 0, 10, ..., 100.

This creates:
    5,450 source images x 11 levels = 59,950 files per factor

Severity 0 is copied byte-for-byte from the clean source image. The other ten
levels are newly corrupted images.

SHARED MANIFEST
---------------
All three programs must use the same --manifest path. The first program creates
it. The other programs load it. This guarantees that every factor and every
severity uses the exact same underlying source images.

EXAMPLE COMMANDS
----------------
Run these commands from the folder containing the four Python files.

1. Motion blur (creates the shared manifest):

python generate_motion_blur_dataset.py \
  --input "/path/to/PlantVillage_clean" \
  --output "/path/to/PlantVillage_motion_blur" \
  --manifest "/path/to/plantvillage_selected_images.csv"

2. Resolution degradation (reuses the manifest):

python generate_resolution_dataset.py \
  --input "/path/to/PlantVillage_clean" \
  --output "/path/to/PlantVillage_resolution" \
  --manifest "/path/to/plantvillage_selected_images.csv"

3. Lighting variation (reuses the manifest):

python generate_lighting_dataset.py \
  --input "/path/to/PlantVillage_clean" \
  --output "/path/to/PlantVillage_lighting" \
  --manifest "/path/to/plantvillage_selected_images.csv"

DEFAULT CORRUPTION SETTINGS
---------------------------
Motion blur:
    angle = 0 degrees
    maximum blur fraction = 0.30
    gamma = 2.0

Resolution:
    minimum scale at severity 100 = 0.03
    upsampling = linear

Lighting:
    minimum multiplier = 0.00
    maximum multiplier = 6.00
    sharpness = 12.0
    pattern = diagonal

OUTPUT EXAMPLES
---------------
Motion:
image (1)_motion_s70_k39_a0.JPG

Resolution:
image (1)_resolution_s70_Sc0p3210.JPG

Lighting:
image (1)_lighting_s70_min0p00_max6p00_sh12p00_diagonal.JPG

The complete relative source structure is preserved under each output root.

RESUMING
--------
Existing images are skipped by default. Therefore, rerunning the same command
resumes an interrupted generation run. Add --overwrite only when you intend to
replace all matching files.

REBUILDING THE SELECTION
------------------------
Only use --rebuild-manifest before starting the first corruption factor. Once
one factor has been generated, do not rebuild the manifest; otherwise the
factors may no longer use the same source-image selection.

CUSTOM SEVERITIES
-----------------
Use:
    --severities "0,10,20,30,40,50,60,70,80,90,100"

The default is already this list.

OUTPUT INDEXES
--------------
Each output root receives a CSV index containing the source path, output path,
severity, and factor-specific parameters.
