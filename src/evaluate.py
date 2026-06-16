"""
Evaluation script for gastrointestinal endoscopy image classification.

This script evaluates a trained model on the test split and generates
reviewer-facing reproducibility outputs:

    - Classification report
    - Confusion matrix CSV
    - Confusion matrix figure
    - ROC curve figure
    - Prediction CSV
    - Evaluation metrics JSON
    - Inference-time summary
    - Parameter and approximate FLOPs summary

Example:
    python src/evaluate.py --model proposed --config config.yaml
    python src/evaluate.py --model efficientnetb1 --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

from data_pipeline import (
    dataset_to_numpy,
    ensure_output_directories,
    get_datasets,
    load_config,
    set_global_seed,
    validate_dataset_structure,
)
from models import (
    SUPPORTED_MODELS,
    build_and_compile_model,
    count_total_parameters,
    count_trainable_parameters,
    estimate_flops_giga,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate GI endoscopy classification models."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="proposed",
        choices=sorted(SUPPORTED_MODELS),
        help="Model to evaluate.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Optional path to a specific .weights.h5 file.",
    )
    parser.add_argument(
        "--saved-model",
        type=str,
        default=None,
        help="Optional path to a specific .keras saved model.",
    )
    return parser.parse_args()


def prepare_directories(config: Dict) -> Dict[str, Path]:
    """Create and return output directories."""
    ensure_output_directories(config)

    paths = config.get("paths", {})
    directories = {
        "output": Path(paths.get("output_dir", "outputs")),
        "checkpoint": Path(paths.get("checkpoint_dir", "outputs/checkpoints")),
        "log": Path(paths.get("log_dir", "outputs/logs")),
        "report": Path(paths.get("report_dir", "outputs/reports")),
        "figure": Path(paths.get("figure_dir", "outputs/figures")),
    }

    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    return directories


def apply_model_specific_training_config(config: Dict, model_name: str) -> Dict:
    """Apply model-specific settings so evaluation matches training."""
    model_cfg = config.get("baseline_models", {}).get(model_name, {})
    training_cfg = config.setdefault("training", {})

    for key in ["optimizer", "learning_rate", "batch_size", "epochs", "dropout_rate"]:
        if key in model_cfg:
            if key == "dropout_rate":
                config.setdefault("model", {})["dropout_rate"] = model_cfg[key]
            else:
                training_cfg[key] = model_cfg[key]

    return config


def load_trained_model(
    model_name: str,
    config: Dict,
    directories: Dict[str, Path],
    weights_path: str | None = None,
    saved_model_path: str | None = None,
) -> tf.keras.Model:
    """
    Load a trained model.

    Priority:
        1. User-provided saved .keras model
        2. User-provided .weights.h5 file
        3. outputs/checkpoints/<model>_final.keras
        4. outputs/checkpoints/<model>_best.weights.h5
    """
    if saved_model_path:
        path = Path(saved_model_path)
        if not path.exists():
            raise FileNotFoundError(f"Saved model file not found: {path}")
        return tf.keras.models.load_model(str(path), compile=False)

    model = build_and_compile_model(model_name, config)

    if weights_path:
        path = Path(weights_path)
        if not path.exists():
            raise FileNotFoundError(f"Weight file not found: {path}")
        model.load_weights(str(path))
        return model

    final_model_path = directories["checkpoint"] / f"{model_name}_final.keras"
    best_weights_path = directories["checkpoint"] / f"{model_name}_best.weights.h5"

    if final_model_path.exists():
        return tf.keras.models.load_model(str(final_model_path), compile=False)

    if best_weights_path.exists():
        model.load_weights(str(best_weights_path))
        return model

    raise FileNotFoundError(
        "No trained model found. Expected one of:\n"
        f"  {final_model_path}\n"
        f"  {best_weights_path}\n"
        "Train the model first using src/train.py."
    )


def predict_with_timing(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
) -> Tuple[np.ndarray, float, float]:
    """
    Generate predictions and measure inference time.

    Returns
    -------
    y_prob:
        Predicted class probabilities.
    total_seconds:
        Total inference duration.
    ms_per_image:
        Average inference time per image in milliseconds.
    """
    total_samples = 0
    for _, labels in dataset:
        total_samples += int(labels.shape[0])

    start_time = time.perf_counter()
    y_prob = model.predict(dataset, verbose=1)
    total_seconds = time.perf_counter() - start_time

    ms_per_image = (total_seconds / max(total_samples, 1)) * 1000.0
    return y_prob, total_seconds, ms_per_image


def save_classification_outputs(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    directories: Dict[str, Path],
) -> Dict[str, Path]:
    """Save classification report, confusion matrix, and predictions."""
    report_dir = directories["report"]

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_path = report_dir / f"{model_name}_classification_report.csv"
    report_df.to_csv(report_path)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_path = report_dir / f"{model_name}_confusion_matrix.csv"
    cm_df.to_csv(cm_path)

    prediction_df = pd.DataFrame(
        {
            "sample_index": np.arange(len(y_true)),
            "true_label_index": y_true,
            "predicted_label_index": y_pred,
            "true_label": [class_names[index] for index in y_true],
            "predicted_label": [class_names[index] for index in y_pred],
            "correct": y_true == y_pred,
            "prediction_confidence": np.max(y_prob, axis=1),
        }
    )

    for index, class_name in enumerate(class_names):
        prediction_df[f"prob_{class_name}"] = y_prob[:, index]

    prediction_path = report_dir / f"{model_name}_predictions.csv"
    prediction_df.to_csv(prediction_path, index=False)

    return {
        "classification_report": report_path,
        "confusion_matrix": cm_path,
        "predictions": prediction_path,
    }


def plot_confusion_matrix(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    directories: Dict[str, Path],
    dpi: int = 300,
) -> Path:
    """Save confusion matrix figure."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm)

    ax.set_title(f"Confusion Matrix - {model_name}")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=35, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    output_path = directories["figure"] / f"{model_name}_confusion_matrix.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_multiclass_roc(
    model_name: str,
    y_true_onehot: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    directories: Dict[str, Path],
    dpi: int = 300,
) -> Tuple[Path, Dict[str, float]]:
    """Save multi-class ROC curve and return class-wise AUC scores."""
    fig, ax = plt.subplots(figsize=(7, 6))

    auc_scores: Dict[str, float] = {}

    for class_index, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, class_index], y_prob[:, class_index])
        roc_auc = auc(fpr, tpr)
        auc_scores[class_name] = float(roc_auc)
        ax.plot(fpr, tpr, label=f"{class_name} (AUC={roc_auc:.4f})")

    ax.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    ax.set_title(f"ROC Curve - {model_name}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, linewidth=0.3)
    fig.tight_layout()

    output_path = directories["figure"] / f"{model_name}_roc_curve.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return output_path, auc_scores


def save_metric_summary(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    auc_scores: Dict[str, float],
    model: tf.keras.Model,
    total_inference_seconds: float,
    ms_per_image: float,
    directories: Dict[str, Path],
) -> Path:
    """Save core evaluation metrics as JSON."""
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    weighted_recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    metrics = {
        "model": model_name,
        "test_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "class_auc": auc_scores,
        "macro_auc": float(np.mean(list(auc_scores.values()))) if auc_scores else None,
        "total_inference_seconds": round(float(total_inference_seconds), 6),
        "inference_time_ms_per_image": round(float(ms_per_image), 6),
        "total_parameters": int(count_total_parameters(model)),
        "trainable_parameters": int(count_trainable_parameters(model)),
        "estimated_gflops_proxy": float(estimate_flops_giga(model)),
    }

    output_path = directories["report"] / f"{model_name}_metrics.json"
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return output_path


def append_model_comparison_row(
    model_name: str,
    metrics_path: Path,
    directories: Dict[str, Path],
) -> Path:
    """Append or update model-level metrics in a comparison CSV."""
    comparison_path = directories["report"] / "model_comparison_summary.csv"

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    row = {
        "model": metrics["model"],
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "macro_auc": metrics["macro_auc"],
        "parameters": metrics["total_parameters"],
        "trainable_parameters": metrics["trainable_parameters"],
        "estimated_gflops_proxy": metrics["estimated_gflops_proxy"],
        "inference_time_ms_per_image": metrics["inference_time_ms_per_image"],
        "test_samples": metrics["test_samples"],
    }

    if comparison_path.exists():
        comparison_df = pd.read_csv(comparison_path)
        comparison_df = comparison_df[comparison_df["model"] != model_name]
        comparison_df = pd.concat([comparison_df, pd.DataFrame([row])], ignore_index=True)
    else:
        comparison_df = pd.DataFrame([row])

    comparison_df = comparison_df.sort_values(by="model").reset_index(drop=True)
    comparison_df.to_csv(comparison_path, index=False)
    return comparison_path


def print_evaluation_summary(model_name: str, metrics_path: Path, output_paths: Dict[str, Path]) -> None:
    """Print concise evaluation summary."""
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    print("\nEvaluation completed.")
    print(f"Model                         : {model_name}")
    print(f"Accuracy                      : {metrics['accuracy']:.6f}")
    print(f"Macro Precision               : {metrics['macro_precision']:.6f}")
    print(f"Macro Recall                  : {metrics['macro_recall']:.6f}")
    print(f"Macro F1-score                : {metrics['macro_f1']:.6f}")
    print(f"Macro ROC-AUC                 : {metrics['macro_auc']:.6f}")
    print(f"Inference time / image (ms)   : {metrics['inference_time_ms_per_image']:.6f}")
    print(f"Total parameters              : {metrics['total_parameters']:,}")
    print(f"Metrics JSON                  : {metrics_path}")

    for key, value in output_paths.items():
        print(f"{key.replace('_', ' ').title():30}: {value}")


def main() -> None:
    """Main evaluation routine."""
    args = parse_args()
    model_name = args.model.lower().strip()

    config = load_config(args.config)
    config = apply_model_specific_training_config(config, model_name)

    project_cfg = config.get("project", {})
    reproducibility_cfg = config.get("reproducibility", {})
    seed = int(project_cfg.get("seed", 42))
    deterministic_ops = bool(reproducibility_cfg.get("deterministic_ops", False))
    set_global_seed(seed, deterministic_ops=deterministic_ops)

    directories = prepare_directories(config)

    print("=" * 70)
    print("GI Endoscopy Image Classification Evaluation")
    print("=" * 70)
    print(f"Model  : {model_name}")
    print(f"Config : {args.config}")
    print("=" * 70)

    summary = validate_dataset_structure(config, strict=True)
    dataset_summary_path = directories["report"] / "dataset_summary.csv"
    summary.to_csv(dataset_summary_path, index=False)

    print("\nDataset summary:")
    print(summary.to_string(index=False))

    _, _, test_ds, class_names = get_datasets(
        config,
        augment_train=False,
        validate=False,
    )

    print("\nLoading trained model...")
    model = load_trained_model(
        model_name=model_name,
        config=config,
        directories=directories,
        weights_path=args.weights,
        saved_model_path=args.saved_model,
    )

    print("\nConverting test labels...")
    _, y_true_onehot = dataset_to_numpy(test_ds)
    y_true = np.argmax(y_true_onehot, axis=1)

    print("\nRunning inference...")
    y_prob, total_inference_seconds, ms_per_image = predict_with_timing(model, test_ds)
    y_pred = np.argmax(y_prob, axis=1)

    output_paths = save_classification_outputs(
        model_name=model_name,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        class_names=class_names,
        directories=directories,
    )

    dpi = int(config.get("plots", {}).get("dpi", 300))

    if config.get("evaluation", {}).get("save_confusion_matrix", True):
        output_paths["confusion_matrix_figure"] = plot_confusion_matrix(
            model_name=model_name,
            y_true=y_true,
            y_pred=y_pred,
            class_names=class_names,
            directories=directories,
            dpi=dpi,
        )

    if config.get("evaluation", {}).get("save_roc_curve", True):
        roc_path, auc_scores = plot_multiclass_roc(
            model_name=model_name,
            y_true_onehot=y_true_onehot,
            y_prob=y_prob,
            class_names=class_names,
            directories=directories,
            dpi=dpi,
        )
        output_paths["roc_curve_figure"] = roc_path
    else:
        auc_scores = {}

    metrics_path = save_metric_summary(
        model_name=model_name,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        auc_scores=auc_scores,
        model=model,
        total_inference_seconds=total_inference_seconds,
        ms_per_image=ms_per_image,
        directories=directories,
    )

    comparison_path = append_model_comparison_row(
        model_name=model_name,
        metrics_path=metrics_path,
        directories=directories,
    )
    output_paths["model_comparison_summary"] = comparison_path

    print_evaluation_summary(model_name, metrics_path, output_paths)


if __name__ == "__main__":
    main()
