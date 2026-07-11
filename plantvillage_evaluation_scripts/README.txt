PLANTVILLAGE CORRUPTION EVALUATION
==================================

FILES
-----
evaluation_common.py
evaluate_motion_blur.py
evaluate_resolution.py
evaluate_lighting.py
requirements.txt

KEEP ALL FOUR PYTHON FILES IN THE SAME FOLDER.

METRICS
-------
The scripts print and save the same metrics used in the previous experiment:

- Accuracy
- Macro-F1
- Macro precision
- Macro recall
- Per-class accuracy
- Per-class precision, recall, and F1
- Confusion matrices
- Maximum softmax confidence
- Correct/incorrect result for each image
- First severity below 80% accuracy
- First severity below 60% accuracy
- Accuracy at maximum severity

METRIC FILES
------------
Each result folder contains:

overall_metrics.csv
severity_metrics.csv
robustness_thresholds.csv
per_class_metrics.csv
per_class_accuracy_by_severity.csv
predictions.csv
confusion_matrix_overall.csv
confusion_matrix_s0.csv
confusion_matrix_s10.csv
...
class_to_index.json

DEPENDENCIES
------------
Install packages in the same Python environment as the trained model:

pip install torch torchvision numpy pillow scikit-learn tqdm

IMPORTANT: PREPROCESSING
------------------------
The input image size and normalization MUST match the model's original
training/evaluation pipeline. The defaults are:

--image-size 256
--resize-mode resize
--normalization none

If your earlier evaluation used ImageNet normalization, add:

--normalization imagenet

If the original input size was 224, add:

--image-size 224

CLASS ORDER
-----------
The model output order must match the dataset class order.

Without --class-map, classes are sorted alphabetically like ImageFolder.

A class-map JSON list looks like:

[
  "Apple___Apple_scab",
  "Apple___Black_rot",
  ...
]

The list position is the output index. A dictionary is also supported:

{
  "Apple___Apple_scab": 0,
  "Apple___Black_rot": 1
}

If the classifier has 39 outputs, the class map must contain all 39 classes,
even if the generated dataset has only 38 folders. Every dataset folder must
appear in the mapping.

MODEL LOADING
-------------
Example model file:

CNN.py

containing:

class CNN(torch.nn.Module):
    def __init__(self, num_classes):
        ...

The loader automatically tries these constructors:

CNN(num_classes)
CNN(num_classes=num_classes)
CNN(n_classes=num_classes)
CNN()

It supports checkpoints saved as:

torch.save(model.state_dict(), path)
torch.save({"model_state_dict": model.state_dict()}, path)
torch.save({"state_dict": model.state_dict()}, path)
torch.save(model, path)

EXAMPLE: MOTION BLUR
--------------------
python evaluate_motion_blur.py \
  --data "/path/to/PlantVillage_motion_blur" \
  --output "/path/to/results/motion_blur" \
  --model-file "/path/to/CNN.py" \
  --model-class CNN \
  --checkpoint "/path/to/model.pth" \
  --class-map "/path/to/class_to_index.json" \
  --image-size 256 \
  --normalization none \
  --batch-size 64 \
  --num-workers 4 \
  --device auto \
  --amp

EXAMPLE: RESOLUTION
-------------------
python evaluate_resolution.py \
  --data "/path/to/PlantVillage_resolution" \
  --output "/path/to/results/resolution" \
  --model-file "/path/to/CNN.py" \
  --model-class CNN \
  --checkpoint "/path/to/model.pth" \
  --class-map "/path/to/class_to_index.json" \
  --image-size 256 \
  --normalization none \
  --batch-size 64 \
  --num-workers 4 \
  --device auto \
  --amp

EXAMPLE: LIGHTING
-----------------
python evaluate_lighting.py \
  --data "/path/to/PlantVillage_lighting" \
  --output "/path/to/results/lighting" \
  --model-file "/path/to/CNN.py" \
  --model-class CNN \
  --checkpoint "/path/to/model.pth" \
  --class-map "/path/to/class_to_index.json" \
  --image-size 256 \
  --normalization none \
  --batch-size 64 \
  --num-workers 4 \
  --device auto \
  --amp

CPU
---
For CPU evaluation, omit --amp and use:

--device cpu

GPU
---
For an NVIDIA GPU:

--device cuda --amp

Start with batch size 64. Increase to 128 if GPU memory allows.

RESUMING
--------
Evaluation results are rewritten each time. The script does not resume halfway
through inference because all predictions are required to compute metrics.
