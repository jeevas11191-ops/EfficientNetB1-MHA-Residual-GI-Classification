"""
Training script for gastrointestinal endoscopy image classification.

This script trains one selected model using the same preprocessing,
augmentation, optimization, and reporting protocol defined in config.yaml.

Example:
    python src/train.py --model proposed --config config.yaml
    python src/train.py --model efficientnetb1 --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from data_pipeline import (
    count_dataset_images,
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
    save_model_summary,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train GI endoscopy classification models."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="proposed",
        choices=sorted(SUPPORTED_MODELS),
        help="Model to train.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml.",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable training-time augmentation for this run.",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run a 2-epoch smoke test without changing config.yaml.",
    )
    return parser.parse_args()


def apply_model_specific_training_config(config: Dict, model_name: str) -> Dict:
    """
    Apply per-model training settings from baseline_models.<model_name>.

    This keeps the comparison reproducible while allowing every compared method
    to report its optimizer, learning rate, batch size, epochs, and dropout
    settings explicitly.
    """
    model_cfg = config.get("baseline_models", {}).get(model_name, {})
    training_cfg = config.setdefault("training", {})

    for key in ["optimizer", "learning_rate", "batch_size", "epochs", "dropout_rate"]:
        if key in model_cfg:
            if key == "dropout_rate":
                config.setdefault("model", {})["dropout_rate"] = model_cfg[key]
            else:
                training_cfg[key] = model_cfg[key]

    return config


def prepare_run_directories(config: Dict) -> Dict[str, Path]:
    """Create and return standard output directories."""
    ensure_output_directories(config)

    paths = config.get("paths", {})
    directories = {
        "output": Path(paths.get("output_dir", "outputs")),
        "checkpoint": Path(paths.get("checkpoint_dir", "outputs/checkpoints")),
        "log": Path(paths.get("log_dir", "outputs/logs")),
        "report": Path(paths.get("report_dir", "outputs/reports")),
        "figure": Path(paths.get("figure_dir", "outputs/figures")),
        "environment": Path(paths.get("output_dir", "outputs")) / "environment",
    }

    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    return directories


def save_config_copy(config_path: str | Path, config: Dict, model_name: str, directories: Dict[str, Path]) -> Path:
    """Save a copy of the configuration used for the run."""
    config_copy_path = directories["report"] / f"{model_name}_config_used.yaml"

    with config_copy_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    original_path = Path(config_path)
    if original_path.exists():
        original_copy_path = directories["report"] / f"{model_name}_original_config_file.yaml"
        shutil.copyfile(original_path, original_copy_path)

    return config_copy_path


def save_environment_summary(model_name: str, directories: Dict[str, Path]) -> Path:
    """Save a concise environment summary for reproducibility."""
    env_path = directories["environment"] / f"{model_name}_environment_summary.txt"

    gpu_devices = tf.config.list_physical_devices("GPU")
    gpu_names = [device.name for device in gpu_devices]

    lines = [
        f"model_name: {model_name}",
        f"python_version: {sys.version.replace(chr(10), ' ')}",
        f"platform: {platform.platform()}",
        f"processor: {platform.processor()}",
        f"tensorflow_version: {tf.__version__}",
        f"numpy_version: {np.__version__}",
        f"gpu_available: {bool(gpu_devices)}",
        f"gpu_devices: {gpu_names}",
    ]

    env_path.write_text("\n".join(lines), encoding="utf-8")
    return env_path


def get_callbacks(model_name: str, config: Dict, directories: Dict[str, Path]) -> List[tf.keras.callbacks.Callback]:
    """Create training callbacks based on config.yaml."""
    callback_cfg = config.get("callbacks", {})
    callbacks: List[tf.keras.callbacks.Callback] = []

    checkpoint_path = directories["checkpoint"] / f"{model_name}_best.weights.h5"
    training_log_path = directories["log"] / f"{model_name}_training_log.csv"

    checkpoint_cfg = callback_cfg.get("model_checkpoint", {})
    if checkpoint_cfg.get("enabled", True):
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor=checkpoint_cfg.get("monitor", "val_accuracy"),
                mode=checkpoint_cfg.get("mode", "max"),
                save_best_only=bool(checkpoint_cfg.get("save_best_only", True)),
                save_weights_only=True,
                verbose=1,
            )
        )

    early_cfg = callback_cfg.get("early_stopping", {})
    if early_cfg.get("enabled", True):
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=early_cfg.get("monitor", "val_loss"),
                patience=int(early_cfg.get("patience", 15)),
                restore_best_weights=bool(early_cfg.get("restore_best_weights", True)),
                verbose=1,
            )
        )

    reduce_cfg = callback_cfg.get("reduce_lr_on_plateau", {})
    if reduce_cfg.get("enabled", True):
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=reduce_cfg.get("monitor", "val_loss"),
                factor=float(reduce_cfg.get("factor", 0.2)),
                patience=int(reduce_cfg.get("patience", 5)),
                min_lr=float(reduce_cfg.get("min_lr", 1e-6)),
                verbose=1,
            )
        )

    csv_cfg = callback_cfg.get("csv_logger", {})
    if csv_cfg.get("enabled", True):
        callbacks.append(
            tf.keras.callbacks.CSVLogger(
                filename=str(training_log_path),
                separator=",",
                append=False,
            )
        )

    return callbacks


def save_hyperparameter_summary(
    model_name: str,
    config: Dict,
    directories: Dict[str, Path],
    augment_train: bool,
) -> Path:
    """
    Save reviewer-facing hyperparameter details.

    The resulting CSV directly supports the manuscript's results section by
    listing optimizer, learning rate, batch size, epochs, dropout, callbacks,
    preprocessing, and augmentation settings.
    """
    dataset_cfg = config.get("dataset", {})
    preprocessing_cfg = config.get("preprocessing", {})
    augmentation_cfg = config.get("augmentation", {})
    training_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    proposed_cfg = config.get("proposed_model", {})
    callback_cfg = config.get("callbacks", {})

    row = {
        "model": model_name,
        "image_size": dataset_cfg.get("image_size", 224),
        "channels": dataset_cfg.get("channels", 3),
        "num_classes": dataset_cfg.get("num_classes", 4),
        "class_names": "; ".join(dataset_cfg.get("class_names", [])),
        "normalization": preprocessing_cfg.get("normalize", True),
        "rescale_value": preprocessing_cfg.get("rescale_value", 1.0 / 255.0),
        "augmentation_enabled": bool(augmentation_cfg.get("enabled", True)) and augment_train,
        "rotation_range": augmentation_cfg.get("rotation_range", 0),
        "height_shift_range": augmentation_cfg.get("height_shift_range", 0),
        "width_shift_range": augmentation_cfg.get("width_shift_range", 0),
        "shear_range": augmentation_cfg.get("shear_range", 0),
        "zoom_range": augmentation_cfg.get("zoom_range", 0),
        "horizontal_flip": augmentation_cfg.get("horizontal_flip", False),
        "vertical_flip": augmentation_cfg.get("vertical_flip", False),
        "brightness_range": str(augmentation_cfg.get("brightness_range", None)),
        "gaussian_noise": augmentation_cfg.get("gaussian_noise", {}).get("enabled", False),
        "gaussian_mean": augmentation_cfg.get("gaussian_noise", {}).get("mean", 0.0),
        "gaussian_stddev": augmentation_cfg.get("gaussian_noise", {}).get("stddev", 0.03),
        "optimizer": training_cfg.get("optimizer", "adam"),
        "learning_rate": training_cfg.get("learning_rate", 1e-4),
        "batch_size": training_cfg.get("batch_size", 16),
        "epochs": training_cfg.get("epochs", 100),
        "loss": training_cfg.get("loss", "categorical_crossentropy"),
        "adaptive_loss_for_proposed": training_cfg.get("use_adaptive_loss_for_proposed", True),
        "adaptive_loss_alpha": training_cfg.get("adaptive_loss_alpha", 0.15),
        "dropout_rate": model_cfg.get("dropout_rate", 0.5),
        "dense_units": model_cfg.get("dense_units", 256),
        "pretrained_weights": model_cfg.get("pretrained_weights", "imagenet"),
        "trainable_backbone": model_cfg.get("trainable_backbone", True),
        "attention_heads": proposed_cfg.get("attention_heads", ""),
        "attention_key_dim": proposed_cfg.get("attention_key_dim", ""),
        "attention_dropout": proposed_cfg.get("attention_dropout", ""),
        "residual_connection": proposed_cfg.get("residual_connection", ""),
        "checkpoint_monitor": callback_cfg.get("model_checkpoint", {}).get("monitor", "val_accuracy"),
        "early_stopping_patience": callback_cfg.get("early_stopping", {}).get("patience", 15),
        "reduce_lr_factor": callback_cfg.get("reduce_lr_on_plateau", {}).get("factor", 0.2),
        "reduce_lr_patience": callback_cfg.get("reduce_lr_on_plateau", {}).get("patience", 5),
    }

    output_path = directories["report"] / f"{model_name}_hyperparameters.csv"
    pd.DataFrame([row]).to_csv(output_path, index=False)
    return output_path


def save_training_history(
    history: tf.keras.callbacks.History,
    model_name: str,
    directories: Dict[str, Path],
) -> Path:
    """Save Keras training history to CSV."""
    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))

    output_path = directories["log"] / f"{model_name}_history.csv"
    history_df.to_csv(output_path, index=False)
    return output_path


def save_training_metrics_json(
    model_name: str,
    history: tf.keras.callbacks.History,
    model: tf.keras.Model,
    elapsed_seconds: float,
    directories: Dict[str, Path],
) -> Path:
    """Save final training summary metrics."""
    history_data = history.history

    def _last_metric(name: str):
        values = history_data.get(name, [])
        return float(values[-1]) if values else None

    def _best_metric(name: str, mode: str = "max"):
        values = history_data.get(name, [])
        if not values:
            return None
        return float(np.max(values) if mode == "max" else np.min(values))

    metrics = {
        "model": model_name,
        "training_time_seconds": round(float(elapsed_seconds), 4),
        "epochs_completed": len(history_data.get("loss", [])),
        "final_train_loss": _last_metric("loss"),
        "final_train_accuracy": _last_metric("accuracy"),
        "final_val_loss": _last_metric("val_loss"),
        "final_val_accuracy": _last_metric("val_accuracy"),
        "best_val_accuracy": _best_metric("val_accuracy", mode="max"),
        "best_val_loss": _best_metric("val_loss", mode="min"),
        "total_parameters": count_total_parameters(model),
        "trainable_parameters": count_trainable_parameters(model),
        "estimated_gflops_proxy": estimate_flops_giga(model),
    }

    output_path = directories["report"] / f"{model_name}_training_metrics.json"
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return output_path


def load_best_weights_if_available(model: tf.keras.Model, model_name: str, directories: Dict[str, Path]) -> bool:
    """Load best checkpoint weights when available."""
    checkpoint_path = directories["checkpoint"] / f"{model_name}_best.weights.h5"
    if checkpoint_path.exists():
        model.load_weights(str(checkpoint_path))
        return True
    return False


def save_final_model(model: tf.keras.Model, model_name: str, directories: Dict[str, Path]) -> Path:
    """Save final model for later evaluation."""
    final_model_path = directories["checkpoint"] / f"{model_name}_final.keras"
    model.save(str(final_model_path), include_optimizer=False)
    return final_model_path


def print_run_summary(
    model_name: str,
    hyperparameter_path: Path,
    history_path: Path,
    metrics_path: Path,
    final_model_path: Path,
) -> None:
    """Print concise run summary."""
    print("\nTraining completed.")
    print(f"Model                     : {model_name}")
    print(f"Hyperparameter summary    : {hyperparameter_path}")
    print(f"Training history          : {history_path}")
    print(f"Training metrics          : {metrics_path}")
    print(f"Saved model               : {final_model_path}")


def main() -> None:
    """Main training routine."""
    args = parse_args()
    model_name = args.model.lower().strip()

    config = load_config(args.config)
    config = apply_model_specific_training_config(config, model_name)

    if args.quick_test:
        config.setdefault("training", {})["epochs"] = 2

    project_cfg = config.get("project", {})
    reproducibility_cfg = config.get("reproducibility", {})

    seed = int(project_cfg.get("seed", 42))
    deterministic_ops = bool(reproducibility_cfg.get("deterministic_ops", False))
    set_global_seed(seed, deterministic_ops=deterministic_ops)

    directories = prepare_run_directories(config)

    print("=" * 70)
    print("GI Endoscopy Image Classification Training")
    print("=" * 70)
    print(f"Model        : {model_name}")
    print(f"Config       : {args.config}")
    print(f"Seed         : {seed}")
    print(f"Augmentation : {not args.no_augment}")
    print("=" * 70)

    summary = validate_dataset_structure(config, strict=True)
    dataset_summary_path = directories["report"] / "dataset_summary.csv"
    summary.to_csv(dataset_summary_path, index=False)

    print("\nDataset summary:")
    print(summary.to_string(index=False))

    save_config_copy(args.config, config, model_name, directories)
    save_environment_summary(model_name, directories)

    train_ds, val_ds, _, class_names = get_datasets(
        config,
        augment_train=not args.no_augment,
        validate=False,
    )

    print("\nClass names:")
    for index, class_name in enumerate(class_names):
        print(f"  {index}: {class_name}")

    print("\nBuilding model...")
    model = build_and_compile_model(model_name, config)
    model.summary()

    model_summary_path = directories["report"] / f"{model_name}_model_summary.txt"
    save_model_summary(model, str(model_summary_path))

    hyperparameter_path = save_hyperparameter_summary(
        model_name=model_name,
        config=config,
        directories=directories,
        augment_train=not args.no_augment,
    )

    callbacks = get_callbacks(model_name, config, directories)

    training_cfg = config.get("training", {})
    epochs = int(training_cfg.get("epochs", 100))

    print("\nStarting training...")
    start_time = time.perf_counter()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    elapsed_seconds = time.perf_counter() - start_time

    best_loaded = load_best_weights_if_available(model, model_name, directories)
    if best_loaded:
        print(f"Loaded best checkpoint weights for {model_name} before final saving.")

    final_model_path = save_final_model(model, model_name, directories)
    history_path = save_training_history(history, model_name, directories)
    metrics_path = save_training_metrics_json(
        model_name=model_name,
        history=history,
        model=model,
        elapsed_seconds=elapsed_seconds,
        directories=directories,
    )

    print_run_summary(
        model_name=model_name,
        hyperparameter_path=hyperparameter_path,
        history_path=history_path,
        metrics_path=metrics_path,
        final_model_path=final_model_path,
    )


if __name__ == "__main__":
    main()
