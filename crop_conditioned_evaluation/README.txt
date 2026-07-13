CROP-CONDITIONED CORRUPTION EVALUATION
=====================================

Purpose
-------
Evaluate the separately generated Apple, Corn, Grape, and Tomato corruption
datasets while distinguishing crop recognition from disease recognition.

The evaluator reports three accuracies at every severity:

1. unrestricted_exact_accuracy
   The normal model output among all 39 classes must match the exact crop and
   disease class.

2. crop_identification_accuracy
   The unrestricted top-1 output only needs to belong to the correct crop. For
   example, predicting Apple scab for an Apple healthy image counts as correct
   crop identification but incorrect exact disease classification.

3. conditioned_disease_accuracy
   The true crop is treated as known. Logits outside that crop are set to
   negative infinity before argmax. An Apple image can therefore only be
   classified as one of the four Apple classes.

Files
-----
evaluation_config.py           Edit all settings here.
crop_evaluation_common.py      Shared model loading, masking, and metrics.
evaluate_motion_blur.py        Evaluate motion blur only.
evaluate_resolution.py         Evaluate resolution only.
evaluate_lighting.py           Evaluate lighting only.

Setup
-----
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Important preprocessing
-----------------------
The supplied defaults match the repository training transform:
    Resize(255), CenterCrop(224), ToTensor(), no normalization

The Flask application instead directly resizes to 224 x 224. To match Flask,
change RESIZE_MODE to "resize". Do not use ImageNet normalization for this
checkpoint.

Run separately
--------------
python evaluate_motion_blur.py
python evaluate_resolution.py
python evaluate_lighting.py

Main outputs
------------
predictions.csv
severity_metrics.csv
overall_metrics.csv
per_crop_severity_metrics.csv
per_class_severity_metrics.csv
robustness_thresholds.csv
crop_confusion_matrix_overall.csv
crop_confusion_matrix_s*.csv
unrestricted_class_confusion_overall.csv
conditioned_confusion_apple_overall.csv
conditioned_confusion_corn_overall.csv
conditioned_confusion_grape_overall.csv
conditioned_confusion_tomato_overall.csv
evaluation_config_snapshot.json

Keep the original unrestricted 39-class metric as the main benchmark. The
crop-conditioned metric is a separate diagnostic showing disease recognition
when crop identity is already known.
