#!/usr/bin/env python3
"""
Shared evaluation engine for severity-controlled PlantVillage corruptions.

Outputs:
- Console metrics by severity
- overall_metrics.csv
- severity_metrics.csv
- robustness_thresholds.csv
- per_class_metrics.csv
- per_class_accuracy_by_severity.csv
- predictions.csv
- confusion_matrix_overall.csv
- confusion_matrix_sXX.csv for every severity
- class_to_index.json

Metrics match the previous experiment:
- Accuracy
- Macro precision
- Macro recall
- Macro-F1
- Per-class accuracy/recall
- Maximum softmax confidence
- Individual correctness
"""

from __future__ import annotations

import argparse
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
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageFile
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}

FACTOR_PATTERNS: Dict[str, re.Pattern[str]] = {
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
    severity: float


def severity_token_to_float(token: str) -> float:
    return float(token.replace("p", "."))


def severity_label(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def parse_triplet(text: str, argument_name: str) -> Tuple[float, float, float]:
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError(
            f"{argument_name} must contain exactly three comma-separated values."
        )
    return tuple(float(part) for part in parts)  # type: ignore[return-value]


def discover_class_names(data_root: Path) -> List[str]:
    class_names = sorted(
        child.name
        for child in data_root.iterdir()
        if child.is_dir()
    )
    if not class_names:
        raise RuntimeError(
            f"No class subfolders were found beneath dataset root: {data_root}"
        )
    return class_names


def load_class_mapping(
    data_root: Path,
    class_map_path: Optional[Path],
) -> Tuple[List[str], Dict[str, int]]:
    """
    Return complete model class names and folder-name-to-model-index mapping.

    Supported class-map JSON formats:
      1. ["Class_A", "Class_B", ...]
      2. {"Class_A": 0, "Class_B": 1, ...}
      3. {"0": "Class_A", "1": "Class_B", ...}

    If no class map is supplied, immediate dataset folders are sorted
    alphabetically, matching torchvision.datasets.ImageFolder behavior.
    """
    dataset_classes = discover_class_names(data_root)

    if class_map_path is None:
        model_classes = dataset_classes
        mapping = {name: index for index, name in enumerate(model_classes)}
        return model_classes, mapping

    with class_map_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        model_classes = [str(name) for name in payload]
    elif isinstance(payload, dict):
        if all(str(key).isdigit() for key in payload):
            indexed = sorted(
                ((int(key), str(value)) for key, value in payload.items()),
                key=lambda pair: pair[0],
            )
            expected = list(range(len(indexed)))
            actual = [index for index, _ in indexed]
            if actual != expected:
                raise ValueError(
                    "Integer-like class-map keys must be contiguous from 0."
                )
            model_classes = [name for _, name in indexed]
        else:
            pairs = [(str(name), int(index)) for name, index in payload.items()]
            pairs.sort(key=lambda pair: pair[1])
            expected = list(range(len(pairs)))
            actual = [index for _, index in pairs]
            if actual != expected:
                raise ValueError(
                    "Class-map values must be contiguous integer indices from 0."
                )
            model_classes = [name for name, _ in pairs]
    else:
        raise ValueError(
            "Class map must be a JSON list or dictionary."
        )

    mapping = {name: index for index, name in enumerate(model_classes)}
    missing = sorted(set(dataset_classes).difference(mapping))
    if missing:
        raise ValueError(
            "Dataset folders missing from class map:\n- "
            + "\n- ".join(missing)
        )

    return model_classes, mapping


def collect_records(
    data_root: Path,
    factor: str,
    class_mapping: Mapping[str, int],
) -> List[ImageRecord]:
    pattern = FACTOR_PATTERNS[factor]
    records: List[ImageRecord] = []
    invalid_filenames: List[Path] = []

    for class_dir in sorted(
        child for child in data_root.iterdir() if child.is_dir()
    ):
        class_name = class_dir.name
        if class_name not in class_mapping:
            raise ValueError(
                f"Class folder is not represented in class mapping: {class_name}"
            )

        for image_path in sorted(class_dir.rglob("*")):
            if (
                not image_path.is_file()
                or image_path.suffix.lower() not in IMAGE_EXTENSIONS
            ):
                continue

            match = pattern.search(image_path.stem)
            if match is None:
                invalid_filenames.append(image_path)
                continue

            records.append(
                ImageRecord(
                    path=image_path,
                    relative_path=image_path.relative_to(data_root),
                    class_name=class_name,
                    class_index=class_mapping[class_name],
                    severity=severity_token_to_float(
                        match.group("severity")
                    ),
                )
            )

    if invalid_filenames:
        examples = "\n".join(
            f"  {path}" for path in invalid_filenames[:10]
        )
        raise ValueError(
            f"Found {len(invalid_filenames)} image file(s) whose names do not "
            f"contain the expected {factor} severity tag. Examples:\n{examples}"
        )

    if not records:
        raise RuntimeError(
            f"No tagged {factor} images were found in: {data_root}"
        )

    records.sort(
        key=lambda record: (
            record.severity,
            record.class_name,
            record.relative_path.as_posix(),
        )
    )
    return records


class CorruptionDataset(Dataset):
    def __init__(
        self,
        records: Sequence[ImageRecord],
        transform: Callable,
    ) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[torch.Tensor, int, float, int]:
        record = self.records[index]
        try:
            with Image.open(record.path) as image:
                rgb_image = image.convert("RGB")
                tensor = self.transform(rgb_image)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load image: {record.path}"
            ) from exc

        return tensor, record.class_index, record.severity, index


def create_transform(args: argparse.Namespace) -> transforms.Compose:
    operations: List[Callable] = []

    if args.resize_mode == "resize":
        operations.append(
            transforms.Resize((args.image_size, args.image_size))
        )
    elif args.resize_mode == "shorter-center-crop":
        operations.extend(
            [
                transforms.Resize(args.resize_shorter),
                transforms.CenterCrop(args.image_size),
            ]
        )
    else:
        raise ValueError(f"Unsupported resize mode: {args.resize_mode}")

    operations.append(transforms.ToTensor())

    if args.normalization == "imagenet":
        operations.append(
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )
        )
    elif args.normalization == "custom":
        mean = parse_triplet(args.mean, "--mean")
        std = parse_triplet(args.std, "--std")
        if any(value <= 0 for value in std):
            raise ValueError("Every --std value must be positive.")
        operations.append(transforms.Normalize(mean=mean, std=std))
    elif args.normalization != "none":
        raise ValueError(
            "--normalization must be none, imagenet, or custom."
        )

    return transforms.Compose(operations)


def import_model_class(
    model_file: Path,
    class_name: str,
):
    module_name = f"_evaluation_model_{model_file.stem}"

    # Allow the model file to import helper modules located beside it.
    model_directory = str(model_file.parent)
    if model_directory not in sys.path:
        sys.path.insert(0, model_directory)

    spec = importlib.util.spec_from_file_location(module_name, model_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import model file: {model_file}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, class_name):
        available = sorted(
            name
            for name, value in vars(module).items()
            if inspect.isclass(value)
        )
        raise AttributeError(
            f"Class {class_name!r} was not found in {model_file}. "
            f"Available classes: {available}"
        )

    return getattr(module, class_name)


def instantiate_model(
    model_class,
    num_classes: int,
    constructor_style: str,
):
    attempts = []

    if constructor_style in {"auto", "positional"}:
        attempts.append(("positional", lambda: model_class(num_classes)))

    if constructor_style in {"auto", "num_classes"}:
        attempts.append(
            ("num_classes keyword", lambda: model_class(num_classes=num_classes))
        )

    if constructor_style in {"auto", "n_classes"}:
        attempts.append(
            ("n_classes keyword", lambda: model_class(n_classes=num_classes))
        )

    if constructor_style in {"auto", "empty"}:
        attempts.append(("empty constructor", lambda: model_class()))

    errors = []
    for description, constructor in attempts:
        try:
            model = constructor()
            print(f"Model constructor used: {description}")
            return model
        except TypeError as exc:
            errors.append(f"{description}: {exc}")

    raise TypeError(
        "Could not instantiate the model class. Constructor attempts:\n- "
        + "\n- ".join(errors)
    )


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, torch.nn.Module):
        return checkpoint.state_dict()

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Checkpoint must be a state dictionary, a dictionary containing "
            "one, or a serialized torch.nn.Module."
        )

    for key in (
        "state_dict",
        "model_state_dict",
        "model",
        "net",
        "network",
    ):
        candidate = checkpoint.get(key)
        if isinstance(candidate, dict):
            return candidate
        if isinstance(candidate, torch.nn.Module):
            return candidate.state_dict()

    if checkpoint and all(
        isinstance(key, str) for key in checkpoint.keys()
    ):
        return checkpoint

    raise ValueError("Unable to locate a model state dictionary.")


def normalize_state_dict_keys(state_dict: Mapping[str, torch.Tensor]):
    prefixes = ("module.", "model.", "_orig_mod.")
    cleaned = dict(state_dict)

    changed = True
    while changed and cleaned:
        changed = False
        for prefix in prefixes:
            if all(key.startswith(prefix) for key in cleaned):
                cleaned = {
                    key[len(prefix):]: value
                    for key, value in cleaned.items()
                }
                changed = True
                break

    return cleaned


def load_model(args: argparse.Namespace, num_classes: int, device: torch.device):
    model_class = import_model_class(
        Path(args.model_file).expanduser().resolve(),
        args.model_class,
    )
    model = instantiate_model(
        model_class,
        num_classes=num_classes,
        constructor_style=args.constructor_style,
    )

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    load_kwargs = {"map_location": "cpu"}

    # PyTorch 2.6 changed torch.load's default to weights_only=True.
    # Explicitly allow locally trusted full-module checkpoints while retaining
    # compatibility with older PyTorch versions that lack this argument.
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = False

    checkpoint = torch.load(checkpoint_path, **load_kwargs)

    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint
        print("Loaded serialized torch.nn.Module from checkpoint.")
    else:
        state_dict = normalize_state_dict_keys(
            extract_state_dict(checkpoint)
        )
        load_result = model.load_state_dict(
            state_dict,
            strict=not args.non_strict_checkpoint,
        )
        if args.non_strict_checkpoint:
            print(
                f"Missing checkpoint keys: {list(load_result.missing_keys)}"
            )
            print(
                f"Unexpected checkpoint keys: "
                f"{list(load_result.unexpected_keys)}"
            )

    model.to(device)
    model.eval()
    return model


def choose_device(requested: str) -> torch.device:
    requested = requested.lower()

    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable.")
        return device

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def metric_row(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    return {
        "images": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
    }


def save_dictionary_rows(
    output_path: Path,
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_confusion_matrix(
    output_path: Path,
    matrix: np.ndarray,
    class_names: Sequence[str],
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\predicted", *class_names])
        for class_name, row in zip(class_names, matrix):
            writer.writerow([class_name, *row.tolist()])


def find_first_below(
    severity_rows: Sequence[Mapping[str, object]],
    threshold: float,
) -> Optional[float]:
    for row in sorted(
        severity_rows,
        key=lambda item: float(item["severity"]),
    ):
        if float(row["accuracy"]) < threshold:
            return float(row["severity"])
    return None


def format_threshold(value: Optional[float]) -> str:
    if value is None:
        return "Not reached"
    return severity_label(value)


def evaluate(
    args: argparse.Namespace,
    factor: str,
) -> None:
    started_at = time.time()

    data_root = Path(args.data).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    model_file = Path(args.model_file).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    class_map_path = (
        Path(args.class_map).expanduser().resolve()
        if args.class_map
        else None
    )

    if not data_root.is_dir():
        raise NotADirectoryError(f"Dataset folder not found: {data_root}")
    if not model_file.is_file():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint file not found: {checkpoint_path}"
        )
    if class_map_path is not None and not class_map_path.is_file():
        raise FileNotFoundError(
            f"Class mapping file not found: {class_map_path}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    class_names, folder_to_index = load_class_mapping(
        data_root,
        class_map_path,
    )
    num_classes = len(class_names)

    if args.num_classes is not None and args.num_classes != num_classes:
        raise ValueError(
            f"--num-classes={args.num_classes}, but the class map contains "
            f"{num_classes} classes. These must match."
        )

    records = collect_records(
        data_root=data_root,
        factor=factor,
        class_mapping=folder_to_index,
    )
    severities = sorted({record.severity for record in records})

    class_counts: Dict[str, Dict[float, int]] = {
        class_name: {severity: 0 for severity in severities}
        for class_name in folder_to_index
        if any(record.class_name == class_name for record in records)
    }
    for record in records:
        class_counts[record.class_name][record.severity] += 1

    expected_per_class = {
        class_name: set(counts.values())
        for class_name, counts in class_counts.items()
    }
    unequal = {
        class_name: sorted(values)
        for class_name, values in expected_per_class.items()
        if len(values) != 1
    }
    if unequal:
        print(
            "WARNING: some classes do not have the same image count at every "
            f"severity: {unequal}"
        )

    transform = create_transform(args)
    dataset = CorruptionDataset(records, transform)

    device = choose_device(args.device)
    pin_memory = device.type == "cuda"
    persistent_workers = args.num_workers > 0

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )

    model = load_model(args, num_classes=num_classes, device=device)

    if device.type == "cuda":
        print(f"Device: CUDA — {torch.cuda.get_device_name(device)}")
    else:
        print(f"Device: {device}")
    print(f"Factor: {factor}")
    print(f"Images: {len(records):,}")
    print(f"Classes in model mapping: {num_classes}")
    print(f"Severity levels: {[severity_label(s) for s in severities]}")
    print(f"Batch size: {args.batch_size}")
    print(f"Workers: {args.num_workers}")
    print(f"Normalization: {args.normalization}")
    print()

    all_true = np.empty(len(records), dtype=np.int64)
    all_pred = np.empty(len(records), dtype=np.int64)
    all_severity = np.empty(len(records), dtype=np.float32)
    all_confidence = np.empty(len(records), dtype=np.float32)

    use_amp = bool(args.amp and device.type == "cuda")
    progress = tqdm(loader, desc=f"Evaluating {factor}", unit="batch")

    with torch.inference_mode():
        for images, labels, severity_values, record_indices in progress:
            images = images.to(
                device,
                non_blocking=pin_memory,
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=use_amp,
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
                    raise TypeError(
                        "Model returned a dictionary without logits/out/output."
                    )

            if outputs.ndim != 2:
                raise ValueError(
                    f"Expected model output shape [batch, classes], got "
                    f"{tuple(outputs.shape)}."
                )
            if outputs.shape[1] != num_classes:
                raise ValueError(
                    f"Model produced {outputs.shape[1]} outputs, but the class "
                    f"mapping contains {num_classes} classes."
                )

            probabilities = torch.softmax(outputs.float(), dim=1)
            confidence, predictions = probabilities.max(dim=1)

            indices_np = record_indices.numpy()
            all_true[indices_np] = labels.numpy()
            all_pred[indices_np] = predictions.cpu().numpy()
            all_severity[indices_np] = severity_values.numpy()
            all_confidence[indices_np] = confidence.cpu().numpy()

    correct = all_true == all_pred

    # Save class mapping.
    with (output_dir / "class_to_index.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {name: index for index, name in enumerate(class_names)},
            handle,
            indent=2,
        )

    # Individual predictions.
    prediction_rows = []
    for index, record in enumerate(records):
        predicted_index = int(all_pred[index])
        predicted_name = (
            class_names[predicted_index]
            if 0 <= predicted_index < len(class_names)
            else f"UNKNOWN_INDEX_{predicted_index}"
        )
        prediction_rows.append(
            {
                "factor": factor,
                "severity": severity_label(float(all_severity[index])),
                "image_path": record.relative_path.as_posix(),
                "true_index": int(all_true[index]),
                "true_class": record.class_name,
                "predicted_index": predicted_index,
                "predicted_class": predicted_name,
                "confidence": float(all_confidence[index]),
                "correct": int(correct[index]),
            }
        )

    save_dictionary_rows(
        output_dir / "predictions.csv",
        prediction_rows,
        fieldnames=[
            "factor",
            "severity",
            "image_path",
            "true_index",
            "true_class",
            "predicted_index",
            "predicted_class",
            "confidence",
            "correct",
        ],
    )

    # Overall metrics, including and excluding the clean severity-0 copies.
    overall_rows = []
    overall_all = metric_row(all_true, all_pred)
    overall_rows.append(
        {
            "factor": factor,
            "scope": "all_severities_including_0",
            **overall_all,
            "mean_confidence": float(all_confidence.mean()),
            "mean_correct_confidence": float(
                all_confidence[correct].mean()
            ) if correct.any() else math.nan,
            "mean_incorrect_confidence": float(
                all_confidence[~correct].mean()
            ) if (~correct).any() else math.nan,
        }
    )

    nonzero_mask = all_severity > 0
    if nonzero_mask.any():
        nonzero_metrics = metric_row(
            all_true[nonzero_mask],
            all_pred[nonzero_mask],
        )
        overall_rows.append(
            {
                "factor": factor,
                "scope": "corrupted_only_severity_gt_0",
                **nonzero_metrics,
                "mean_confidence": float(
                    all_confidence[nonzero_mask].mean()
                ),
                "mean_correct_confidence": float(
                    all_confidence[nonzero_mask & correct].mean()
                ) if (nonzero_mask & correct).any() else math.nan,
                "mean_incorrect_confidence": float(
                    all_confidence[nonzero_mask & ~correct].mean()
                ) if (nonzero_mask & ~correct).any() else math.nan,
            }
        )

    save_dictionary_rows(
        output_dir / "overall_metrics.csv",
        overall_rows,
        fieldnames=[
            "factor",
            "scope",
            "images",
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "mean_confidence",
            "mean_correct_confidence",
            "mean_incorrect_confidence",
        ],
    )

    # Severity-level metrics.
    severity_rows = []
    for severity in severities:
        mask = np.isclose(all_severity, severity)
        values = metric_row(all_true[mask], all_pred[mask])
        severity_rows.append(
            {
                "factor": factor,
                "severity": severity,
                **values,
                "mean_confidence": float(all_confidence[mask].mean()),
                "mean_correct_confidence": float(
                    all_confidence[mask & correct].mean()
                ) if (mask & correct).any() else math.nan,
                "mean_incorrect_confidence": float(
                    all_confidence[mask & ~correct].mean()
                ) if (mask & ~correct).any() else math.nan,
            }
        )

    save_dictionary_rows(
        output_dir / "severity_metrics.csv",
        severity_rows,
        fieldnames=[
            "factor",
            "severity",
            "images",
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "mean_confidence",
            "mean_correct_confidence",
            "mean_incorrect_confidence",
        ],
    )

    # Robustness thresholds.
    max_severity_row = max(
        severity_rows,
        key=lambda row: float(row["severity"]),
    )
    threshold_rows = [
        {
            "factor": factor,
            "first_severity_below_80_accuracy": format_threshold(
                find_first_below(severity_rows, 0.80)
            ),
            "first_severity_below_60_accuracy": format_threshold(
                find_first_below(severity_rows, 0.60)
            ),
            "maximum_tested_severity": severity_label(
                float(max_severity_row["severity"])
            ),
            "accuracy_at_maximum_severity": float(
                max_severity_row["accuracy"]
            ),
        }
    ]
    save_dictionary_rows(
        output_dir / "robustness_thresholds.csv",
        threshold_rows,
        fieldnames=[
            "factor",
            "first_severity_below_80_accuracy",
            "first_severity_below_60_accuracy",
            "maximum_tested_severity",
            "accuracy_at_maximum_severity",
        ],
    )

    # Per-class metrics by severity.
    per_class_rows = []
    labels = np.arange(num_classes)

    for severity in severities:
        mask = np.isclose(all_severity, severity)
        y_true_severity = all_true[mask]
        y_pred_severity = all_pred[mask]
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true_severity,
            y_pred_severity,
            labels=labels,
            zero_division=0,
        )

        for class_index, class_name in enumerate(class_names):
            class_mask = y_true_severity == class_index
            class_accuracy = (
                float(
                    np.mean(
                        y_pred_severity[class_mask] == class_index
                    )
                )
                if class_mask.any()
                else math.nan
            )
            per_class_rows.append(
                {
                    "factor": factor,
                    "severity": severity,
                    "class_index": class_index,
                    "class_name": class_name,
                    "support": int(support[class_index]),
                    "accuracy": class_accuracy,
                    "precision": float(precision[class_index]),
                    "recall": float(recall[class_index]),
                    "f1": float(f1[class_index]),
                }
            )

    save_dictionary_rows(
        output_dir / "per_class_metrics.csv",
        per_class_rows,
        fieldnames=[
            "factor",
            "severity",
            "class_index",
            "class_name",
            "support",
            "accuracy",
            "precision",
            "recall",
            "f1",
        ],
    )

    # Heatmap-ready wide table: one row per class, one column per severity.
    accuracy_lookup = {
        (int(row["class_index"]), float(row["severity"])): row["accuracy"]
        for row in per_class_rows
    }
    heatmap_rows = []
    severity_columns = [
        f"severity_{severity_label(severity)}"
        for severity in severities
    ]
    for class_index, class_name in enumerate(class_names):
        row = {
            "class_index": class_index,
            "class_name": class_name,
        }
        for severity, column in zip(severities, severity_columns):
            row[column] = accuracy_lookup.get(
                (class_index, severity),
                math.nan,
            )
        heatmap_rows.append(row)

    save_dictionary_rows(
        output_dir / "per_class_accuracy_by_severity.csv",
        heatmap_rows,
        fieldnames=["class_index", "class_name", *severity_columns],
    )

    # Confusion matrices.
    overall_cm = confusion_matrix(
        all_true,
        all_pred,
        labels=labels,
    )
    save_confusion_matrix(
        output_dir / "confusion_matrix_overall.csv",
        overall_cm,
        class_names,
    )

    for severity in severities:
        mask = np.isclose(all_severity, severity)
        matrix = confusion_matrix(
            all_true[mask],
            all_pred[mask],
            labels=labels,
        )
        save_confusion_matrix(
            output_dir
            / f"confusion_matrix_s{severity_label(severity)}.csv",
            matrix,
            class_names,
        )

    # Console report.
    print("\n" + "=" * 88)
    print(
        f"{'Severity':>10} {'Images':>10} {'Accuracy':>12} "
        f"{'Macro-F1':>12} {'Precision':>12} {'Recall':>12}"
    )
    print("-" * 88)
    for row in severity_rows:
        print(
            f"{severity_label(float(row['severity'])):>10} "
            f"{int(row['images']):>10,} "
            f"{100 * float(row['accuracy']):>11.2f}% "
            f"{100 * float(row['macro_f1']):>11.2f}% "
            f"{100 * float(row['macro_precision']):>11.2f}% "
            f"{100 * float(row['macro_recall']):>11.2f}%"
        )
    print("=" * 88)

    print("\nOverall:")
    for row in overall_rows:
        print(
            f"  {row['scope']}: "
            f"accuracy={100 * float(row['accuracy']):.2f}%, "
            f"macro-F1={100 * float(row['macro_f1']):.2f}%, "
            f"precision={100 * float(row['macro_precision']):.2f}%, "
            f"recall={100 * float(row['macro_recall']):.2f}%"
        )

    threshold = threshold_rows[0]
    print("\nRobustness thresholds:")
    print(
        "  First severity below 80% accuracy: "
        f"{threshold['first_severity_below_80_accuracy']}"
    )
    print(
        "  First severity below 60% accuracy: "
        f"{threshold['first_severity_below_60_accuracy']}"
    )
    print(
        "  Accuracy at maximum severity: "
        f"{100 * float(threshold['accuracy_at_maximum_severity']):.2f}%"
    )

    elapsed = time.time() - started_at
    print(f"\nEvaluation completed in {elapsed / 60:.2f} minutes.")
    print(f"Results folder: {output_dir}")


def build_parser(factor: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            f"Evaluate the {factor} PlantVillage corruption dataset by "
            "severity using a PyTorch classifier."
        )
    )
    parser.add_argument(
        "--data",
        required=True,
        help=(
            "Root of the generated corruption dataset. Immediate subfolders "
            "must be class folders."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Folder in which evaluation CSV files will be written.",
    )
    parser.add_argument(
        "--model-file",
        required=True,
        help=(
            "Python file defining the model class, such as CNN.py."
        ),
    )
    parser.add_argument(
        "--model-class",
        default="CNN",
        help="Model class name inside --model-file. Default: CNN.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the trained PyTorch checkpoint.",
    )
    parser.add_argument(
        "--class-map",
        default=None,
        help=(
            "Optional JSON model class order. Strongly recommended when the "
            "model was not trained with alphabetical ImageFolder ordering."
        ),
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help=(
            "Optional validation value. It must equal the class-map size."
        ),
    )
    parser.add_argument(
        "--constructor-style",
        choices=[
            "auto",
            "positional",
            "num_classes",
            "n_classes",
            "empty",
        ],
        default="auto",
        help="How to instantiate the model class. Default: auto.",
    )
    parser.add_argument(
        "--non-strict-checkpoint",
        action="store_true",
        help=(
            "Load checkpoint with strict=False and report missing/unexpected "
            "keys. Use only when intentionally needed."
        ),
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Final square image size supplied to the CNN. Default: 256.",
    )
    parser.add_argument(
        "--resize-mode",
        choices=["resize", "shorter-center-crop"],
        default="resize",
        help=(
            "Resize directly to image-size, or resize shorter edge and center "
            "crop. Default: resize."
        ),
    )
    parser.add_argument(
        "--resize-shorter",
        type=int,
        default=256,
        help=(
            "Short-edge resize used only with shorter-center-crop. "
            "Default: 256."
        ),
    )
    parser.add_argument(
        "--normalization",
        choices=["none", "imagenet", "custom"],
        default="none",
        help=(
            "Pixel normalization after ToTensor. Use the exact preprocessing "
            "used during model training. Default: none."
        ),
    )
    parser.add_argument(
        "--mean",
        default="0.485,0.456,0.406",
        help="Custom RGB means when --normalization custom.",
    )
    parser.add_argument(
        "--std",
        default="0.229,0.224,0.225",
        help="Custom RGB standard deviations when --normalization custom.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Inference batch size. Default: 64.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Parallel DataLoader workers. Default: 4.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="Batches prefetched per worker. Default: 2.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help=(
            "auto, cpu, cuda, cuda:0, or mps. Default: auto."
        ),
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help=(
            "Use CUDA float16 autocasting for faster inference."
        ),
    )
    return parser


def run_factor(factor: str) -> None:
    parser = build_parser(factor)
    args = parser.parse_args()
    evaluate(args, factor=factor)
