#!/usr/bin/env python3
"""Evaluate crop-only corruptions with unrestricted and crop-conditioned outputs.

Three core accuracies are reported:
1. unrestricted_exact_accuracy:
   Correct disease class among all 39 model outputs.
2. crop_identification_accuracy:
   Whether the unrestricted top-1 class belongs to the correct crop, even if
   the disease class is wrong.
3. conditioned_disease_accuracy:
   Correct disease class after masking all outputs outside the known true crop.
"""

from __future__ import annotations

import csv
import importlib.util
import inspect
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageFile
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

import evaluation_config as config


ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}
FACTOR_PATTERNS = {
    "motion_blur": re.compile(
        r"_motion_s(?P<severity>\d+(?:p\d+)?)", re.IGNORECASE
    ),
    "resolution": re.compile(
        r"_resolution_s(?P<severity>\d+(?:p\d+)?)", re.IGNORECASE
    ),
    "lighting": re.compile(
        r"_lighting_s(?P<severity>\d+(?:p\d+)?)", re.IGNORECASE
    ),
}


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    relative_path: Path
    class_name: str
    class_index: int
    crop_name: str
    crop_index: int
    severity: float


def severity_from_token(token: str) -> float:
    return float(token.replace("p", "."))


def severity_label(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value).replace(".", "p")


def factor_paths(factor: str) -> Tuple[Path, Path]:
    mapping = {
        "motion_blur": (config.MOTION_BLUR_DATA, config.MOTION_BLUR_RESULTS),
        "resolution": (config.RESOLUTION_DATA, config.RESOLUTION_RESULTS),
        "lighting": (config.LIGHTING_DATA, config.LIGHTING_RESULTS),
    }
    try:
        data, output = mapping[factor]
    except KeyError as exc:
        raise ValueError(f"Unsupported factor: {factor}") from exc
    return Path(data).expanduser().resolve(), Path(output).expanduser().resolve()


def validate_crop_groups(
    crop_groups: Mapping[str, Sequence[str]],
) -> Tuple[List[str], Dict[str, str]]:
    crop_names = list(crop_groups.keys())
    if not crop_names:
        raise ValueError("CROP_CLASS_GROUPS cannot be empty.")
    class_to_crop: Dict[str, str] = {}
    for crop_name, class_names in crop_groups.items():
        if not class_names:
            raise ValueError(f"Crop {crop_name!r} has no classes.")
        for class_name in class_names:
            if class_name in class_to_crop:
                raise ValueError(
                    f"Class {class_name!r} appears in multiple crop groups."
                )
            class_to_crop[class_name] = crop_name
    return crop_names, class_to_crop


def load_class_mapping(path: Path) -> Tuple[List[str], Dict[str, int]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        class_names = [str(value) for value in payload]
    elif isinstance(payload, dict):
        if all(str(key).isdigit() for key in payload.keys()):
            pairs = sorted(
                ((int(key), str(value)) for key, value in payload.items()),
                key=lambda item: item[0],
            )
            if [index for index, _ in pairs] != list(range(len(pairs))):
                raise ValueError("Class-map keys must be contiguous from 0.")
            class_names = [name for _, name in pairs]
        else:
            pairs = sorted(
                ((str(name), int(index)) for name, index in payload.items()),
                key=lambda item: item[1],
            )
            if [index for _, index in pairs] != list(range(len(pairs))):
                raise ValueError("Class-map values must be contiguous from 0.")
            class_names = [name for name, _ in pairs]
    else:
        raise ValueError("Class map must be a JSON list or dictionary.")

    return class_names, {name: index for index, name in enumerate(class_names)}


def collect_records(
    data_root: Path,
    factor: str,
    class_to_index: Mapping[str, int],
    crop_names: Sequence[str],
    class_to_crop: Mapping[str, str],
) -> List[ImageRecord]:
    pattern = FACTOR_PATTERNS[factor]
    crop_to_index = {name: index for index, name in enumerate(crop_names)}
    records: List[ImageRecord] = []
    invalid: List[Path] = []

    class_dirs = sorted(path for path in data_root.iterdir() if path.is_dir())
    if not class_dirs:
        raise RuntimeError(f"No class folders found under: {data_root}")

    for class_dir in class_dirs:
        class_name = class_dir.name
        if class_name not in class_to_index:
            raise ValueError(f"Dataset class missing from class map: {class_name}")
        if class_name not in class_to_crop:
            raise ValueError(
                f"Dataset class is not in CROP_CLASS_GROUPS: {class_name}"
            )
        crop_name = class_to_crop[class_name]
        for image_path in sorted(class_dir.rglob("*")):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            match = pattern.search(image_path.stem)
            if match is None:
                invalid.append(image_path)
                continue
            records.append(
                ImageRecord(
                    path=image_path,
                    relative_path=image_path.relative_to(data_root),
                    class_name=class_name,
                    class_index=class_to_index[class_name],
                    crop_name=crop_name,
                    crop_index=crop_to_index[crop_name],
                    severity=severity_from_token(match.group("severity")),
                )
            )

    if invalid:
        examples = "\n".join(str(path) for path in invalid[:10])
        raise ValueError(
            f"Found {len(invalid)} files without the expected {factor} severity "
            f"tag. Examples:\n{examples}"
        )
    if not records:
        raise RuntimeError(f"No tagged {factor} images found in: {data_root}")
    records.sort(
        key=lambda row: (
            row.severity,
            row.crop_name,
            row.class_name,
            row.relative_path.as_posix(),
        )
    )
    return records


class CropCorruptionDataset(Dataset):
    def __init__(self, records: Sequence[ImageRecord], transform: Callable):
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        try:
            with Image.open(record.path) as image:
                tensor = self.transform(image.convert("RGB"))
        except Exception as exc:
            raise RuntimeError(f"Failed to load image: {record.path}") from exc
        return (
            tensor,
            record.class_index,
            record.crop_index,
            record.severity,
            index,
        )


def create_transform() -> transforms.Compose:
    operations: List[Callable] = []
    if config.RESIZE_MODE == "resize":
        operations.append(transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)))
    elif config.RESIZE_MODE == "shorter-center-crop":
        operations.extend(
            [
                transforms.Resize(config.RESIZE_SHORTER),
                transforms.CenterCrop(config.IMAGE_SIZE),
            ]
        )
    else:
        raise ValueError(
            "RESIZE_MODE must be 'resize' or 'shorter-center-crop'."
        )
    operations.append(transforms.ToTensor())

    if config.NORMALIZATION == "imagenet":
        operations.append(
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )
        )
    elif config.NORMALIZATION == "custom":
        if len(config.CUSTOM_MEAN) != 3 or len(config.CUSTOM_STD) != 3:
            raise ValueError("CUSTOM_MEAN and CUSTOM_STD must each have 3 values.")
        if any(float(value) <= 0 for value in config.CUSTOM_STD):
            raise ValueError("Every CUSTOM_STD value must be positive.")
        operations.append(
            transforms.Normalize(
                mean=tuple(float(value) for value in config.CUSTOM_MEAN),
                std=tuple(float(value) for value in config.CUSTOM_STD),
            )
        )
    elif config.NORMALIZATION != "none":
        raise ValueError("NORMALIZATION must be none, imagenet, or custom.")
    return transforms.Compose(operations)


def import_model_class(model_file: Path, class_name: str):
    model_directory = str(model_file.parent)
    if model_directory not in sys.path:
        sys.path.insert(0, model_directory)
    module_name = f"_crop_eval_model_{model_file.stem}"
    spec = importlib.util.spec_from_file_location(module_name, model_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import model file: {model_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, class_name):
        raise AttributeError(f"Model class {class_name!r} not found in {model_file}")
    return getattr(module, class_name)


def instantiate_model(model_class):
    style = str(config.CONSTRUCTOR_STYLE)
    attempts = []
    if style in {"auto", "positional"}:
        attempts.append(("positional", lambda: model_class(config.NUM_CLASSES)))
    if style in {"auto", "num_classes"}:
        attempts.append(
            ("num_classes keyword", lambda: model_class(num_classes=config.NUM_CLASSES))
        )
    if style in {"auto", "n_classes"}:
        attempts.append(
            ("n_classes keyword", lambda: model_class(n_classes=config.NUM_CLASSES))
        )
    if style in {"auto", "empty"}:
        attempts.append(("empty", lambda: model_class()))

    errors = []
    for description, constructor in attempts:
        try:
            model = constructor()
            print(f"Model constructor used: {description}")
            return model
        except TypeError as exc:
            errors.append(f"{description}: {exc}")
    raise TypeError("Could not instantiate model:\n- " + "\n- ".join(errors))


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, torch.nn.Module):
        return checkpoint.state_dict()
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint is not a state dictionary or model.")
    for key in ("state_dict", "model_state_dict", "model", "net", "network"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
        if isinstance(value, torch.nn.Module):
            return value.state_dict()
    if checkpoint and all(isinstance(key, str) for key in checkpoint):
        return checkpoint
    raise ValueError("Unable to locate a state dictionary in checkpoint.")


def clean_state_dict_keys(state_dict):
    cleaned = dict(state_dict)
    prefixes = ("module.", "model.", "_orig_mod.")
    changed = True
    while changed and cleaned:
        changed = False
        for prefix in prefixes:
            if all(key.startswith(prefix) for key in cleaned):
                cleaned = {key[len(prefix):]: value for key, value in cleaned.items()}
                changed = True
                break
    return cleaned


def load_model(device: torch.device):
    model_file = Path(config.MODEL_FILE).expanduser().resolve()
    checkpoint_path = Path(config.CHECKPOINT).expanduser().resolve()
    model_class = import_model_class(model_file, config.MODEL_CLASS)
    model = instantiate_model(model_class)

    kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    checkpoint = torch.load(checkpoint_path, **kwargs)
    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint
    else:
        result = model.load_state_dict(
            clean_state_dict_keys(extract_state_dict(checkpoint)),
            strict=bool(config.STRICT_CHECKPOINT),
        )
        if not config.STRICT_CHECKPOINT:
            print(f"Missing keys: {list(result.missing_keys)}")
            print(f"Unexpected keys: {list(result.unexpected_keys)}")
    model.to(device)
    model.eval()
    return model


def choose_device() -> torch.device:
    requested = str(config.DEVICE).lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable.")
    return device


def save_rows(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_matrix(path: Path, matrix: np.ndarray, row_names, column_names):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\predicted", *column_names])
        for name, row in zip(row_names, matrix):
            writer.writerow([name, *row.tolist()])


def rectangular_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    true_labels: Sequence[int],
    predicted_labels: Sequence[int],
) -> np.ndarray:
    row_lookup = {label: index for index, label in enumerate(true_labels)}
    col_lookup = {label: index for index, label in enumerate(predicted_labels)}
    matrix = np.zeros((len(true_labels), len(predicted_labels)), dtype=np.int64)
    for true_value, predicted_value in zip(y_true, y_pred):
        if true_value in row_lookup and predicted_value in col_lookup:
            matrix[row_lookup[int(true_value)], col_lookup[int(predicted_value)]] += 1
    return matrix


def safe_macro_f1(y_true, y_pred, labels) -> float:
    if len(y_true) == 0:
        return math.nan
    return float(
        f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0)
    )


def metrics_for_mask(
    mask: np.ndarray,
    true_class: np.ndarray,
    unrestricted_pred: np.ndarray,
    true_crop: np.ndarray,
    predicted_crop: np.ndarray,
    conditioned_pred: np.ndarray,
    target_class_indices: Sequence[int],
    crop_indices: Sequence[int],
) -> Dict[str, float | int]:
    count = int(mask.sum())
    if count == 0:
        return {
            "images": 0,
            "unrestricted_exact_accuracy": math.nan,
            "crop_identification_accuracy": math.nan,
            "conditioned_disease_accuracy": math.nan,
            "unrestricted_exact_macro_f1": math.nan,
            "crop_identification_macro_f1": math.nan,
            "conditioned_disease_macro_f1": math.nan,
        }
    return {
        "images": count,
        "unrestricted_exact_accuracy": float(
            np.mean(unrestricted_pred[mask] == true_class[mask])
        ),
        "crop_identification_accuracy": float(
            np.mean(predicted_crop[mask] == true_crop[mask])
        ),
        "conditioned_disease_accuracy": float(
            np.mean(conditioned_pred[mask] == true_class[mask])
        ),
        "unrestricted_exact_macro_f1": safe_macro_f1(
            true_class[mask], unrestricted_pred[mask], target_class_indices
        ),
        "crop_identification_macro_f1": safe_macro_f1(
            true_crop[mask], predicted_crop[mask], crop_indices
        ),
        "conditioned_disease_macro_f1": safe_macro_f1(
            true_class[mask], conditioned_pred[mask], target_class_indices
        ),
    }


def first_below(rows, field: str, threshold: float) -> Optional[float]:
    for row in sorted(rows, key=lambda item: float(item["severity"])):
        value = float(row[field])
        if value < threshold:
            return float(row["severity"])
    return None


def snapshot_config(output_dir: Path, factor: str, data_root: Path) -> None:
    if not config.SAVE_CONFIG_SNAPSHOT:
        return
    payload = {
        "factor": factor,
        "data_root": str(data_root),
        "model_file": str(Path(config.MODEL_FILE).expanduser().resolve()),
        "checkpoint": str(Path(config.CHECKPOINT).expanduser().resolve()),
        "class_map": str(Path(config.CLASS_MAP).expanduser().resolve()),
        "model_class": config.MODEL_CLASS,
        "num_classes": config.NUM_CLASSES,
        "constructor_style": config.CONSTRUCTOR_STYLE,
        "strict_checkpoint": config.STRICT_CHECKPOINT,
        "image_size": config.IMAGE_SIZE,
        "resize_mode": config.RESIZE_MODE,
        "resize_shorter": config.RESIZE_SHORTER,
        "normalization": config.NORMALIZATION,
        "custom_mean": list(config.CUSTOM_MEAN),
        "custom_std": list(config.CUSTOM_STD),
        "device": config.DEVICE,
        "batch_size": config.BATCH_SIZE,
        "num_workers": config.NUM_WORKERS,
        "prefetch_factor": config.PREFETCH_FACTOR,
        "use_amp": config.USE_AMP,
        "crop_class_groups": config.CROP_CLASS_GROUPS,
    }
    with (output_dir / "evaluation_config_snapshot.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, indent=2)


def evaluate_factor(factor: str) -> None:
    started = time.time()
    data_root, output_dir = factor_paths(factor)
    model_file = Path(config.MODEL_FILE).expanduser().resolve()
    checkpoint = Path(config.CHECKPOINT).expanduser().resolve()
    class_map_path = Path(config.CLASS_MAP).expanduser().resolve()

    for path, description, expected in [
        (data_root, "dataset folder", "dir"),
        (model_file, "model file", "file"),
        (checkpoint, "checkpoint", "file"),
        (class_map_path, "class map", "file"),
    ]:
        exists = path.is_dir() if expected == "dir" else path.is_file()
        if not exists:
            raise FileNotFoundError(f"{description.title()} not found: {path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    class_names, class_to_index = load_class_mapping(class_map_path)
    if len(class_names) != int(config.NUM_CLASSES):
        raise ValueError(
            f"Class map contains {len(class_names)} classes, but NUM_CLASSES="
            f"{config.NUM_CLASSES}."
        )

    crop_names, class_to_crop = validate_crop_groups(config.CROP_CLASS_GROUPS)
    missing_from_map = sorted(set(class_to_crop).difference(class_to_index))
    if missing_from_map:
        raise ValueError(
            "Configured crop classes missing from class map:\n- "
            + "\n- ".join(missing_from_map)
        )
    crop_to_index = {name: index for index, name in enumerate(crop_names)}
    other_crop_index = len(crop_names)
    crop_output_names = [*crop_names, "Other_or_Background"]

    allowed_indices_by_crop = {
        crop_to_index[crop]: torch.tensor(
            [class_to_index[name] for name in class_list], dtype=torch.long
        )
        for crop, class_list in config.CROP_CLASS_GROUPS.items()
    }
    target_class_names = [
        class_name
        for crop in crop_names
        for class_name in config.CROP_CLASS_GROUPS[crop]
    ]
    target_class_indices = [class_to_index[name] for name in target_class_names]

    records = collect_records(
        data_root,
        factor,
        class_to_index,
        crop_names,
        class_to_crop,
    )
    severities = sorted({record.severity for record in records})
    dataset = CropCorruptionDataset(records, create_transform())
    device = choose_device()
    pin_memory = device.type == "cuda"
    loader = DataLoader(
        dataset,
        batch_size=int(config.BATCH_SIZE),
        shuffle=False,
        num_workers=int(config.NUM_WORKERS),
        pin_memory=pin_memory,
        persistent_workers=int(config.NUM_WORKERS) > 0,
        prefetch_factor=(
            int(config.PREFETCH_FACTOR) if int(config.NUM_WORKERS) > 0 else None
        ),
    )
    model = load_model(device)

    print(f"Device: {device}")
    print(f"Factor: {factor}")
    print(f"Images: {len(records):,}")
    print(f"Target disease classes: {len(target_class_names)}")
    print(f"Crops: {crop_names}")
    print(f"Severities: {[severity_label(value) for value in severities]}")
    print(f"Preprocessing: {config.RESIZE_MODE}, {config.NORMALIZATION}\n")

    n = len(records)
    true_class = np.empty(n, dtype=np.int64)
    true_crop = np.empty(n, dtype=np.int64)
    severity_array = np.empty(n, dtype=np.float32)
    unrestricted_pred = np.empty(n, dtype=np.int64)
    unrestricted_conf = np.empty(n, dtype=np.float32)
    predicted_crop = np.empty(n, dtype=np.int64)
    conditioned_pred = np.empty(n, dtype=np.int64)
    conditioned_conf = np.empty(n, dtype=np.float32)

    use_amp = bool(config.USE_AMP and device.type == "cuda")
    with torch.inference_mode():
        for images, labels, crop_indices, severity_values, record_indices in tqdm(
            loader, desc=f"Evaluating {factor}", unit="batch"
        ):
            images = images.to(device, non_blocking=pin_memory)
            labels_device = labels.to(device)
            crop_indices_device = crop_indices.to(device)
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=use_amp
            ):
                outputs = model(images)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
            elif isinstance(outputs, dict):
                for key in ("logits", "out", "output"):
                    if key in outputs:
                        outputs = outputs[key]
                        break
                else:
                    raise TypeError("Model dictionary output lacks logits/out/output.")
            if outputs.ndim != 2 or outputs.shape[1] != len(class_names):
                raise ValueError(
                    f"Expected [batch, {len(class_names)}] logits, got "
                    f"{tuple(outputs.shape)}."
                )

            probabilities = torch.softmax(outputs.float(), dim=1)
            unrestricted_conf_batch, unrestricted_pred_batch = probabilities.max(dim=1)

            predicted_crop_batch = torch.full(
                (outputs.shape[0],),
                other_crop_index,
                dtype=torch.long,
                device=device,
            )
            for crop_index_value, allowed_cpu in allowed_indices_by_crop.items():
                allowed = allowed_cpu.to(device)
                belongs = (unrestricted_pred_batch[:, None] == allowed[None, :]).any(dim=1)
                predicted_crop_batch[belongs] = crop_index_value

            masked_outputs = torch.full_like(outputs, float("-inf"))
            for crop_index_value, allowed_cpu in allowed_indices_by_crop.items():
                row_mask = crop_indices_device == crop_index_value
                if row_mask.any():
                    allowed = allowed_cpu.to(device)
                    row_indices = torch.where(row_mask)[0]
                    masked_outputs[
                        row_indices[:, None], allowed[None, :]
                    ] = outputs[row_indices[:, None], allowed[None, :]]

            conditioned_probabilities = torch.softmax(masked_outputs.float(), dim=1)
            conditioned_conf_batch, conditioned_pred_batch = (
                conditioned_probabilities.max(dim=1)
            )

            indices = record_indices.numpy()
            true_class[indices] = labels.numpy()
            true_crop[indices] = crop_indices.numpy()
            severity_array[indices] = severity_values.numpy()
            unrestricted_pred[indices] = unrestricted_pred_batch.cpu().numpy()
            unrestricted_conf[indices] = unrestricted_conf_batch.cpu().numpy()
            predicted_crop[indices] = predicted_crop_batch.cpu().numpy()
            conditioned_pred[indices] = conditioned_pred_batch.cpu().numpy()
            conditioned_conf[indices] = conditioned_conf_batch.cpu().numpy()

    unrestricted_correct = unrestricted_pred == true_class
    crop_correct = predicted_crop == true_crop
    conditioned_correct = conditioned_pred == true_class

    snapshot_config(output_dir, factor, data_root)
    with (output_dir / "class_to_index.json").open("w", encoding="utf-8") as handle:
        json.dump(class_to_index, handle, indent=2)
    with (output_dir / "crop_class_groups.json").open("w", encoding="utf-8") as handle:
        json.dump(config.CROP_CLASS_GROUPS, handle, indent=2)

    prediction_rows = []
    for index, record in enumerate(records):
        up = int(unrestricted_pred[index])
        cp = int(conditioned_pred[index])
        predicted_crop_name = crop_output_names[int(predicted_crop[index])]
        prediction_rows.append(
            {
                "factor": factor,
                "severity": severity_label(float(severity_array[index])),
                "image_path": record.relative_path.as_posix(),
                "true_crop": record.crop_name,
                "true_index": int(true_class[index]),
                "true_class": record.class_name,
                "unrestricted_predicted_index": up,
                "unrestricted_predicted_class": class_names[up],
                "unrestricted_predicted_crop": predicted_crop_name,
                "unrestricted_confidence": float(unrestricted_conf[index]),
                "unrestricted_exact_correct": int(unrestricted_correct[index]),
                "crop_identification_correct": int(crop_correct[index]),
                "conditioned_predicted_index": cp,
                "conditioned_predicted_class": class_names[cp],
                "conditioned_confidence_within_crop": float(conditioned_conf[index]),
                "conditioned_disease_correct": int(conditioned_correct[index]),
            }
        )
    save_rows(
        output_dir / "predictions.csv",
        prediction_rows,
        list(prediction_rows[0].keys()),
    )

    crop_indices_for_metrics = list(range(len(crop_names)))
    all_mask = np.ones(n, dtype=bool)
    overall_metrics = metrics_for_mask(
        all_mask,
        true_class,
        unrestricted_pred,
        true_crop,
        predicted_crop,
        conditioned_pred,
        target_class_indices,
        crop_indices_for_metrics,
    )
    nonzero_mask = severity_array > 0
    overall_rows = [
        {"factor": factor, "scope": "all_severities_including_0", **overall_metrics}
    ]
    if nonzero_mask.any():
        overall_rows.append(
            {
                "factor": factor,
                "scope": "corrupted_only_severity_gt_0",
                **metrics_for_mask(
                    nonzero_mask,
                    true_class,
                    unrestricted_pred,
                    true_crop,
                    predicted_crop,
                    conditioned_pred,
                    target_class_indices,
                    crop_indices_for_metrics,
                ),
            }
        )
    save_rows(
        output_dir / "overall_metrics.csv",
        overall_rows,
        list(overall_rows[0].keys()),
    )

    severity_rows = []
    for severity in severities:
        mask = np.isclose(severity_array, severity)
        severity_rows.append(
            {
                "factor": factor,
                "severity": severity,
                **metrics_for_mask(
                    mask,
                    true_class,
                    unrestricted_pred,
                    true_crop,
                    predicted_crop,
                    conditioned_pred,
                    target_class_indices,
                    crop_indices_for_metrics,
                ),
            }
        )
    save_rows(
        output_dir / "severity_metrics.csv",
        severity_rows,
        list(severity_rows[0].keys()),
    )

    per_crop_rows = []
    for severity in severities:
        severity_mask = np.isclose(severity_array, severity)
        for crop_index, crop_name in enumerate(crop_names):
            mask = severity_mask & (true_crop == crop_index)
            crop_class_indices = [
                class_to_index[name] for name in config.CROP_CLASS_GROUPS[crop_name]
            ]
            per_crop_rows.append(
                {
                    "factor": factor,
                    "severity": severity,
                    "crop": crop_name,
                    **metrics_for_mask(
                        mask,
                        true_class,
                        unrestricted_pred,
                        true_crop,
                        predicted_crop,
                        conditioned_pred,
                        crop_class_indices,
                        [crop_index],
                    ),
                }
            )
    save_rows(
        output_dir / "per_crop_severity_metrics.csv",
        per_crop_rows,
        list(per_crop_rows[0].keys()),
    )

    per_class_rows = []
    for severity in severities:
        severity_mask = np.isclose(severity_array, severity)
        for class_name in target_class_names:
            class_index = class_to_index[class_name]
            crop_name = class_to_crop[class_name]
            mask = severity_mask & (true_class == class_index)
            count = int(mask.sum())
            per_class_rows.append(
                {
                    "factor": factor,
                    "severity": severity,
                    "crop": crop_name,
                    "class_index": class_index,
                    "class_name": class_name,
                    "support": count,
                    "unrestricted_exact_accuracy": (
                        float(np.mean(unrestricted_correct[mask])) if count else math.nan
                    ),
                    "crop_identification_accuracy": (
                        float(np.mean(crop_correct[mask])) if count else math.nan
                    ),
                    "conditioned_disease_accuracy": (
                        float(np.mean(conditioned_correct[mask])) if count else math.nan
                    ),
                }
            )
    save_rows(
        output_dir / "per_class_severity_metrics.csv",
        per_class_rows,
        list(per_class_rows[0].keys()),
    )

    threshold_rows = []
    for metric_name in (
        "unrestricted_exact_accuracy",
        "crop_identification_accuracy",
        "conditioned_disease_accuracy",
    ):
        max_row = max(severity_rows, key=lambda row: float(row["severity"]))
        below_80 = first_below(severity_rows, metric_name, 0.80)
        below_60 = first_below(severity_rows, metric_name, 0.60)
        threshold_rows.append(
            {
                "factor": factor,
                "metric": metric_name,
                "first_severity_below_80": (
                    "Not reached" if below_80 is None else severity_label(below_80)
                ),
                "first_severity_below_60": (
                    "Not reached" if below_60 is None else severity_label(below_60)
                ),
                "maximum_tested_severity": severity_label(
                    float(max_row["severity"])
                ),
                "value_at_maximum_severity": float(max_row[metric_name]),
            }
        )
    save_rows(
        output_dir / "robustness_thresholds.csv",
        threshold_rows,
        list(threshold_rows[0].keys()),
    )

    crop_matrix = confusion_matrix(
        true_crop,
        predicted_crop,
        labels=list(range(len(crop_output_names))),
    )
    save_matrix(
        output_dir / "crop_confusion_matrix_overall.csv",
        crop_matrix,
        crop_output_names,
        crop_output_names,
    )
    for severity in severities:
        mask = np.isclose(severity_array, severity)
        matrix = confusion_matrix(
            true_crop[mask],
            predicted_crop[mask],
            labels=list(range(len(crop_output_names))),
        )
        save_matrix(
            output_dir / f"crop_confusion_matrix_s{severity_label(severity)}.csv",
            matrix,
            crop_output_names,
            crop_output_names,
        )

    unrestricted_matrix = rectangular_confusion(
        true_class,
        unrestricted_pred,
        target_class_indices,
        list(range(len(class_names))),
    )
    save_matrix(
        output_dir / "unrestricted_class_confusion_overall.csv",
        unrestricted_matrix,
        target_class_names,
        class_names,
    )

    for crop_name in crop_names:
        crop_class_names = list(config.CROP_CLASS_GROUPS[crop_name])
        crop_class_indices = [class_to_index[name] for name in crop_class_names]
        crop_mask = true_crop == crop_to_index[crop_name]
        matrix = confusion_matrix(
            true_class[crop_mask],
            conditioned_pred[crop_mask],
            labels=crop_class_indices,
        )
        save_matrix(
            output_dir / f"conditioned_confusion_{crop_name.lower()}_overall.csv",
            matrix,
            crop_class_names,
            crop_class_names,
        )

    print("\n" + "=" * 105)
    print(
        f"{'Severity':>9} {'Images':>9} {'Exact 39-class':>17} "
        f"{'Crop ID':>13} {'Disease | crop known':>22}"
    )
    print("-" * 105)
    for row in severity_rows:
        print(
            f"{severity_label(float(row['severity'])):>9} "
            f"{int(row['images']):>9,} "
            f"{100 * float(row['unrestricted_exact_accuracy']):>16.2f}% "
            f"{100 * float(row['crop_identification_accuracy']):>12.2f}% "
            f"{100 * float(row['conditioned_disease_accuracy']):>21.2f}%"
        )
    print("=" * 105)
    print("\nInterpretation:")
    print("  Exact 39-class: crop and disease must both be correct.")
    print("  Crop ID: unrestricted top prediction only needs the correct crop.")
    print("  Disease | crop known: nonmatching crop logits are masked first.")
    print(f"\nEvaluation completed in {(time.time() - started) / 60:.2f} minutes.")
    print(f"Results folder: {output_dir}")
