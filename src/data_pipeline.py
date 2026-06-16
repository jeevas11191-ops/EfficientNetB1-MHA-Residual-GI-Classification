"""
Data loading, preprocessing, augmentation, and dataset validation utilities.

This module supports the reproducible gastrointestinal endoscopy image
classification pipeline used by the EfficientNetB1 + Multi-Head Attention +
Residual Connections framework.

Expected dataset layout:

data/
├── train/
│   ├── normal/
│   ├── ulcerative_colitis/
│   ├── polyps/
│   └── esophagitis/
├── val/
│   ├── normal/
│   ├── ulcerative_colitis/
│   ├── polyps/
│   └── esophagitis/
└── test/
    ├── normal/
    ├── ulcerative_colitis/
    ├── polyps/
    └── esophagitis/
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml


ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_config(config_path: str | Path) -> Dict:
    """Load YAML configuration file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration file is empty or invalid.")

    return config


def set_global_seed(seed: int = 42, deterministic_ops: bool = False) -> None:
    """Set seeds for reproducible experiments."""
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic_ops:
        os.environ["TF_DETERMINISTIC_OPS"] = "1"

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_output_directories(config: Dict) -> None:
    """Create required output directories."""
    paths = config.get("paths", {})
    for key in ["output_dir", "checkpoint_dir", "log_dir", "report_dir", "figure_dir"]:
        directory = paths.get(key)
        if directory:
            Path(directory).mkdir(parents=True, exist_ok=True)


def get_class_names(config: Dict) -> list[str]:
    """Return class names from configuration."""
    class_names = config.get("dataset", {}).get("class_names")
    if not class_names:
        raise ValueError("dataset.class_names is missing in config.yaml")
    return list(class_names)


def _count_images_in_directory(directory: Path) -> int:
    """Count supported image files in a directory."""
    if not directory.exists():
        return 0

    return sum(
        1
        for item in directory.iterdir()
        if item.is_file() and item.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS
    )


def count_dataset_images(config: Dict) -> pd.DataFrame:
    """Count images for each split and class."""
    paths = config.get("paths", {})
    class_names = get_class_names(config)

    split_dirs = {
        "train": Path(paths.get("train_dir", "data/train")),
        "val": Path(paths.get("val_dir", "data/val")),
        "test": Path(paths.get("test_dir", "data/test")),
    }

    rows = []
    for split_name, split_dir in split_dirs.items():
        for class_name in class_names:
            class_dir = split_dir / class_name
            rows.append(
                {
                    "split": split_name,
                    "class_name": class_name,
                    "directory": str(class_dir),
                    "image_count": _count_images_in_directory(class_dir),
                }
            )

    return pd.DataFrame(rows)


def validate_dataset_structure(config: Dict, strict: bool = True) -> pd.DataFrame:
    """
    Validate dataset folders and return image counts.

    Parameters
    ----------
    config:
        Loaded YAML configuration.
    strict:
        If True, raise an error when required folders or images are missing.
    """
    paths = config.get("paths", {})
    class_names = get_class_names(config)

    split_dirs = {
        "train": Path(paths.get("train_dir", "data/train")),
        "val": Path(paths.get("val_dir", "data/val")),
        "test": Path(paths.get("test_dir", "data/test")),
    }

    missing = []
    for split_name, split_dir in split_dirs.items():
        if not split_dir.exists():
            missing.append(str(split_dir))
            continue

        for class_name in class_names:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                missing.append(str(class_dir))

    summary = count_dataset_images(config)

    empty_rows = summary[summary["image_count"] == 0]
    if strict and (missing or not empty_rows.empty):
        message_parts = []
        if missing:
            message_parts.append("Missing folders:\n" + "\n".join(missing))
        if not empty_rows.empty:
            message_parts.append(
                "Folders without readable images:\n"
                + empty_rows[["split", "class_name", "directory"]].to_string(index=False)
            )
        raise FileNotFoundError("\n\n".join(message_parts))

    return summary


def save_dataset_summary(config: Dict, summary: pd.DataFrame) -> Path:
    """Save dataset split/class image-count summary."""
    report_dir = Path(config.get("paths", {}).get("report_dir", "outputs/reports"))
    report_dir.mkdir(parents=True, exist_ok=True)

    output_path = report_dir / "dataset_summary.csv"
    summary.to_csv(output_path, index=False)
    return output_path


def _load_image_dataset(
    directory: str | Path,
    config: Dict,
    split_name: str,
    shuffle: bool,
) -> tf.data.Dataset:
    """Load a split using Keras image_dataset_from_directory."""
    dataset_cfg = config.get("dataset", {})
    training_cfg = config.get("training", {})
    project_cfg = config.get("project", {})
    preprocessing_cfg = config.get("preprocessing", {})

    image_size = int(dataset_cfg.get("image_size", 224))
    batch_size = int(training_cfg.get("batch_size", 16))
    seed = int(project_cfg.get("seed", 42))
    color_mode = dataset_cfg.get("color_mode", "rgb")
    class_mode = dataset_cfg.get("class_mode", "categorical")
    class_names = get_class_names(config)

    dataset = tf.keras.utils.image_dataset_from_directory(
        directory=directory,
        labels="inferred",
        label_mode=class_mode,
        class_names=class_names,
        color_mode=color_mode,
        batch_size=batch_size,
        image_size=(image_size, image_size),
        shuffle=shuffle,
        seed=seed if shuffle else None,
    )

    dataset = dataset.map(
        lambda images, labels: preprocess_batch(images, labels, config),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if split_name == "train" and preprocessing_cfg.get("shuffle_train", True):
        dataset = dataset.shuffle(buffer_size=512, seed=seed, reshuffle_each_iteration=True)

    return dataset


def preprocess_batch(
    images: tf.Tensor,
    labels: tf.Tensor,
    config: Dict,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Apply RGB consistency, float conversion, and normalization."""
    preprocessing_cfg = config.get("preprocessing", {})

    images = tf.cast(images, tf.float32)

    if preprocessing_cfg.get("normalize", True):
        rescale_value = float(preprocessing_cfg.get("rescale_value", 1.0 / 255.0))
        images = images * rescale_value

    images = tf.clip_by_value(images, 0.0, 1.0)
    return images, labels


def _image_projective_transform(
    image: tf.Tensor,
    transform: tf.Tensor,
    fill_mode: str = "CONSTANT",
) -> tf.Tensor:
    """Apply a projective transform to a single image."""
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    transform = tf.cast(tf.reshape(transform, [1, 8]), tf.float32)

    output_shape = tf.shape(image)[0:2]
    image_4d = tf.expand_dims(image, axis=0)

    transformed = tf.raw_ops.ImageProjectiveTransformV3(
        images=image_4d,
        transforms=transform,
        output_shape=output_shape,
        interpolation="BILINEAR",
        fill_mode=fill_mode.upper(),
        fill_value=0.0,
    )

    return tf.squeeze(transformed, axis=0)


def random_shear(image: tf.Tensor, shear_range: float, fill_mode: str = "CONSTANT") -> tf.Tensor:
    """
    Apply random x-axis shear to a single image using TensorFlow projective transform.

    The implementation avoids additional dependencies and keeps the pipeline fully
    reproducible within TensorFlow.
    """
    if shear_range <= 0:
        return image

    image = tf.convert_to_tensor(image, dtype=tf.float32)

    shear = tf.random.uniform(shape=[], minval=-shear_range, maxval=shear_range)
    height = tf.cast(tf.shape(image)[0], tf.float32)
    center_y = height / 2.0

    transform = tf.stack(
        [
            1.0,
            -shear,
            shear * center_y,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    return _image_projective_transform(image, transform, fill_mode=fill_mode)


def add_gaussian_noise(
    image: tf.Tensor,
    mean: float = 0.0,
    stddev: float = 0.03,
) -> tf.Tensor:
    """Apply Gaussian noise and clip image values to [0, 1]."""
    noise = tf.random.normal(shape=tf.shape(image), mean=mean, stddev=stddev, dtype=tf.float32)
    image = image + noise
    return tf.clip_by_value(image, 0.0, 1.0)


def random_brightness_factor(image: tf.Tensor, brightness_range: Iterable[float]) -> tf.Tensor:
    """Apply multiplicative brightness adjustment."""
    brightness_values = list(brightness_range)
    if len(brightness_values) != 2:
        return image

    lower = float(brightness_values[0])
    upper = float(brightness_values[1])

    factor = tf.random.uniform(shape=[], minval=lower, maxval=upper)
    image = image * factor
    return tf.clip_by_value(image, 0.0, 1.0)


def build_augmentation_model(config: Dict) -> tf.keras.Sequential:
    """Build Keras augmentation layers for batch-level transformations."""
    aug_cfg = config.get("augmentation", {})
    fill_mode = aug_cfg.get("fill_mode", "constant")

    rotation_range = float(aug_cfg.get("rotation_range", 0.0)) / 360.0
    height_shift = float(aug_cfg.get("height_shift_range", 0.0))
    width_shift = float(aug_cfg.get("width_shift_range", 0.0))
    zoom_range = float(aug_cfg.get("zoom_range", 0.0))

    layers = []

    if rotation_range > 0:
        layers.append(
            tf.keras.layers.RandomRotation(
                factor=rotation_range,
                fill_mode=fill_mode,
                name="random_rotation",
            )
        )

    if height_shift > 0 or width_shift > 0:
        layers.append(
            tf.keras.layers.RandomTranslation(
                height_factor=height_shift,
                width_factor=width_shift,
                fill_mode=fill_mode,
                name="random_translation",
            )
        )

    if zoom_range > 0:
        layers.append(
            tf.keras.layers.RandomZoom(
                height_factor=(-zoom_range, zoom_range),
                width_factor=(-zoom_range, zoom_range),
                fill_mode=fill_mode,
                name="random_zoom",
            )
        )

    horizontal_flip = bool(aug_cfg.get("horizontal_flip", False))
    vertical_flip = bool(aug_cfg.get("vertical_flip", False))

    if horizontal_flip and vertical_flip:
        layers.append(tf.keras.layers.RandomFlip("horizontal_and_vertical", name="random_flip"))
    elif horizontal_flip:
        layers.append(tf.keras.layers.RandomFlip("horizontal", name="random_flip"))
    elif vertical_flip:
        layers.append(tf.keras.layers.RandomFlip("vertical", name="random_flip"))

    return tf.keras.Sequential(layers, name="training_augmentation")


def apply_augmentation(
    dataset: tf.data.Dataset,
    config: Dict,
) -> tf.data.Dataset:
    """Apply augmentation only to the training dataset."""
    aug_cfg = config.get("augmentation", {})

    if not aug_cfg.get("enabled", True):
        return dataset

    augmentation_model = build_augmentation_model(config)
    shear_range = float(aug_cfg.get("shear_range", 0.0))
    fill_mode = aug_cfg.get("fill_mode", "constant")

    gaussian_cfg = aug_cfg.get("gaussian_noise", {})
    use_gaussian = bool(gaussian_cfg.get("enabled", False))
    gaussian_mean = float(gaussian_cfg.get("mean", 0.0))
    gaussian_stddev = float(gaussian_cfg.get("stddev", 0.03))

    brightness_range = aug_cfg.get("brightness_range", None)

    def augment_batch(images: tf.Tensor, labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        images = augmentation_model(images, training=True)

        if brightness_range is not None:
            images = tf.map_fn(
                lambda img: random_brightness_factor(img, brightness_range),
                images,
                fn_output_signature=tf.float32,
            )

        if shear_range > 0:
            images = tf.map_fn(
                lambda img: random_shear(img, shear_range, fill_mode=fill_mode),
                images,
                fn_output_signature=tf.float32,
            )

        if use_gaussian:
            images = tf.map_fn(
                lambda img: add_gaussian_noise(img, mean=gaussian_mean, stddev=gaussian_stddev),
                images,
                fn_output_signature=tf.float32,
            )

        images = tf.clip_by_value(images, 0.0, 1.0)
        return images, labels

    return dataset.map(augment_batch, num_parallel_calls=tf.data.AUTOTUNE)


def optimize_dataset(dataset: tf.data.Dataset, cache: bool = False) -> tf.data.Dataset:
    """Apply optional cache and prefetch for efficient training."""
    if cache:
        dataset = dataset.cache()
    return dataset.prefetch(tf.data.AUTOTUNE)


def get_datasets(
    config: Dict,
    augment_train: bool = True,
    validate: bool = True,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, list[str]]:
    """
    Create train, validation, and test datasets.

    Returns
    -------
    train_ds, val_ds, test_ds, class_names
    """
    project_cfg = config.get("project", {})
    reproducibility_cfg = config.get("reproducibility", {})
    paths = config.get("paths", {})

    seed = int(project_cfg.get("seed", 42))
    deterministic_ops = bool(reproducibility_cfg.get("deterministic_ops", False))
    set_global_seed(seed, deterministic_ops=deterministic_ops)

    ensure_output_directories(config)

    if validate:
        summary = validate_dataset_structure(config, strict=True)
        save_dataset_summary(config, summary)

    train_dir = paths.get("train_dir", "data/train")
    val_dir = paths.get("val_dir", "data/val")
    test_dir = paths.get("test_dir", "data/test")

    train_ds = _load_image_dataset(train_dir, config, split_name="train", shuffle=True)
    val_ds = _load_image_dataset(val_dir, config, split_name="val", shuffle=False)
    test_ds = _load_image_dataset(test_dir, config, split_name="test", shuffle=False)

    if augment_train:
        train_ds = apply_augmentation(train_ds, config)

    train_ds = optimize_dataset(train_ds, cache=False)
    val_ds = optimize_dataset(val_ds, cache=False)
    test_ds = optimize_dataset(test_ds, cache=False)

    return train_ds, val_ds, test_ds, get_class_names(config)


def dataset_to_numpy(dataset: tf.data.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a dataset into NumPy arrays.

    This is intended for evaluation utilities such as ROC computation and should
    only be used when the test set comfortably fits in memory.
    """
    image_batches = []
    label_batches = []

    for images, labels in dataset:
        image_batches.append(images.numpy())
        label_batches.append(labels.numpy())

    if not image_batches:
        raise ValueError("Cannot convert an empty dataset to NumPy arrays.")

    images_np = np.concatenate(image_batches, axis=0)
    labels_np = np.concatenate(label_batches, axis=0)
    return images_np, labels_np


if __name__ == "__main__":
    cfg = load_config("config.yaml")
    summary_df = validate_dataset_structure(cfg, strict=False)
    ensure_output_directories(cfg)
    output_csv = save_dataset_summary(cfg, summary_df)

    print("Dataset summary:")
    print(summary_df.to_string(index=False))
    print(f"Saved summary to: {output_csv}")
