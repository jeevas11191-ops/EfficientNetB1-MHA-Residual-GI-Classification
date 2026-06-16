"""
Model definitions for gastrointestinal endoscopy image classification.

This module contains the compact baseline models and the proposed
EfficientNetB1 + Multi-Head Attention + Residual Connections architecture.

Implemented models:
    - cnn
    - vgg16
    - resnet50
    - efficientnetb1
    - proposed

All model builders accept images normalized to [0, 1] by data_pipeline.py.
Application-specific preprocessing is handled inside each model when required.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import tensorflow as tf


SUPPORTED_MODELS = {
    "cnn",
    "vgg16",
    "resnet50",
    "efficientnetb1",
    "proposed",
}


def get_input_shape(config: Dict) -> Tuple[int, int, int]:
    """Return input shape from config."""
    model_cfg = config.get("model", {})
    input_shape = model_cfg.get("input_shape", [224, 224, 3])

    if len(input_shape) != 3:
        raise ValueError("model.input_shape must contain [height, width, channels].")

    return tuple(int(value) for value in input_shape)


def get_num_classes(config: Dict) -> int:
    """Return number of output classes."""
    return int(config.get("dataset", {}).get("num_classes", 4))


def get_dropout_rate(config: Dict, model_name: str | None = None) -> float:
    """Return dropout rate for a model."""
    if model_name:
        baseline_cfg = config.get("baseline_models", {}).get(model_name, {})
        if "dropout_rate" in baseline_cfg:
            return float(baseline_cfg["dropout_rate"])

    return float(config.get("model", {}).get("dropout_rate", 0.5))


def get_dense_units(config: Dict) -> int:
    """Return classifier dense units."""
    return int(config.get("model", {}).get("dense_units", 256))


def get_pretrained_weights(config: Dict, model_name: str) -> str | None:
    """Return pretrained weight setting for transfer-learning models."""
    baseline_cfg = config.get("baseline_models", {}).get(model_name, {})
    value = baseline_cfg.get(
        "pretrained_weights",
        config.get("model", {}).get("pretrained_weights", "imagenet"),
    )
    if value in [None, "none", "None", "null"]:
        return None
    return str(value)


def is_backbone_trainable(config: Dict, model_name: str) -> bool:
    """Return whether the transfer-learning backbone is trainable."""
    baseline_cfg = config.get("baseline_models", {}).get(model_name, {})
    return bool(baseline_cfg.get("trainable_backbone", config.get("model", {}).get("trainable_backbone", True)))


def multiply_by_255_layer(name: str = "scale_to_255") -> tf.keras.layers.Layer:
    """Layer that converts [0, 1] images to [0, 255] scale."""
    return tf.keras.layers.Lambda(lambda x: x * 255.0, name=name)


def build_simple_cnn(config: Dict) -> tf.keras.Model:
    """
    Build a lightweight CNN baseline.

    This baseline is intentionally compact and provides a conventional
    non-transfer-learning comparison under the same input resolution.
    """
    input_shape = get_input_shape(config)
    num_classes = get_num_classes(config)
    dropout_rate = get_dropout_rate(config, "cnn")
    dense_units = get_dense_units(config)

    inputs = tf.keras.Input(shape=input_shape, name="input_image")

    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu", name="conv1")(inputs)
    x = tf.keras.layers.BatchNormalization(name="bn1")(x)
    x = tf.keras.layers.MaxPooling2D(name="pool1")(x)

    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu", name="conv2")(x)
    x = tf.keras.layers.BatchNormalization(name="bn2")(x)
    x = tf.keras.layers.MaxPooling2D(name="pool2")(x)

    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu", name="conv3")(x)
    x = tf.keras.layers.BatchNormalization(name="bn3")(x)
    x = tf.keras.layers.MaxPooling2D(name="pool3")(x)

    x = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu", name="conv4")(x)
    x = tf.keras.layers.BatchNormalization(name="bn4")(x)

    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dense(dense_units, activation="relu", name="dense_features")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="classification_head")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="CNN_Baseline")


def build_vgg16(config: Dict) -> tf.keras.Model:
    """Build VGG16 transfer-learning baseline."""
    input_shape = get_input_shape(config)
    num_classes = get_num_classes(config)
    dropout_rate = get_dropout_rate(config, "vgg16")
    dense_units = get_dense_units(config)
    weights = get_pretrained_weights(config, "vgg16")

    inputs = tf.keras.Input(shape=input_shape, name="input_image")
    x = multiply_by_255_layer()(inputs)
    x = tf.keras.applications.vgg16.preprocess_input(x)

    backbone = tf.keras.applications.VGG16(
        include_top=False,
        weights=weights,
        input_tensor=x,
    )
    backbone.trainable = is_backbone_trainable(config, "vgg16")

    x = backbone.output
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dense(dense_units, activation="relu", name="dense_features")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="classification_head")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="VGG16_Baseline")


def build_resnet50(config: Dict) -> tf.keras.Model:
    """Build ResNet50 transfer-learning baseline."""
    input_shape = get_input_shape(config)
    num_classes = get_num_classes(config)
    dropout_rate = get_dropout_rate(config, "resnet50")
    dense_units = get_dense_units(config)
    weights = get_pretrained_weights(config, "resnet50")

    inputs = tf.keras.Input(shape=input_shape, name="input_image")
    x = multiply_by_255_layer()(inputs)
    x = tf.keras.applications.resnet50.preprocess_input(x)

    backbone = tf.keras.applications.ResNet50(
        include_top=False,
        weights=weights,
        input_tensor=x,
    )
    backbone.trainable = is_backbone_trainable(config, "resnet50")

    x = backbone.output
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dense(dense_units, activation="relu", name="dense_features")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="classification_head")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="ResNet50_Baseline")


def build_efficientnetb1_baseline(config: Dict) -> tf.keras.Model:
    """
    Build EfficientNetB1 baseline without attention or residual refinement.

    Dataset images are normalized to [0, 1]. They are scaled back to [0, 255]
    before entering EfficientNetB1 so that pretrained ImageNet statistics remain
    consistent with the original application model.
    """
    input_shape = get_input_shape(config)
    num_classes = get_num_classes(config)
    dropout_rate = get_dropout_rate(config, "efficientnetb1")
    dense_units = get_dense_units(config)
    weights = get_pretrained_weights(config, "efficientnetb1")

    inputs = tf.keras.Input(shape=input_shape, name="input_image")
    x = multiply_by_255_layer()(inputs)

    backbone = tf.keras.applications.EfficientNetB1(
        include_top=False,
        weights=weights,
        input_tensor=x,
    )
    backbone.trainable = is_backbone_trainable(config, "efficientnetb1")

    x = backbone.output
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dense(dense_units, activation="relu", name="dense_features")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="classification_head")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="EfficientNetB1_Baseline")


def spatial_token_reshape(feature_map: tf.Tensor) -> tf.Tensor:
    """
    Reshape a final convolutional feature map into spatial tokens.

    For EfficientNetB1 at 224 x 224 input size, the expected shape is:
        B x 7 x 7 x 1280  ->  B x 49 x 1280

    The function remains dynamic and works with any spatial size.
    """
    shape = tf.shape(feature_map)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]
    channels = shape[3]
    tokens = tf.reshape(feature_map, (batch_size, height * width, channels))
    return tokens


def build_proposed_efficientnetb1_mha_residual(config: Dict) -> tf.keras.Model:
    """
    Build the proposed EfficientNetB1 + Multi-Head Attention + Residual model.

    Pipeline:
        Input image
        -> EfficientNetB1 backbone without classifier
        -> final convolutional feature map
        -> spatial token reshaping
        -> multi-head self-attention
        -> residual addition
        -> layer normalization
        -> global average pooling over tokens
        -> dropout
        -> dense softmax classifier
    """
    input_shape = get_input_shape(config)
    num_classes = get_num_classes(config)
    dropout_rate = get_dropout_rate(config, "proposed")
    dense_units = get_dense_units(config)

    proposed_cfg = config.get("proposed_model", {})
    weights = proposed_cfg.get("pretrained_weights", get_pretrained_weights(config, "proposed"))
    if weights in [None, "none", "None", "null"]:
        weights = None

    attention_heads = int(proposed_cfg.get("attention_heads", 4))
    attention_key_dim = int(proposed_cfg.get("attention_key_dim", 64))
    attention_dropout = float(proposed_cfg.get("attention_dropout", 0.1))
    use_residual = bool(proposed_cfg.get("residual_connection", True))

    inputs = tf.keras.Input(shape=input_shape, name="input_image")
    x = multiply_by_255_layer()(inputs)

    backbone = tf.keras.applications.EfficientNetB1(
        include_top=False,
        weights=weights,
        input_tensor=x,
    )
    backbone.trainable = bool(proposed_cfg.get("trainable_backbone", is_backbone_trainable(config, "proposed")))

    feature_map = backbone.output

    tokens = tf.keras.layers.Lambda(spatial_token_reshape, name="spatial_token_reshape")(feature_map)

    attention_output = tf.keras.layers.MultiHeadAttention(
        num_heads=attention_heads,
        key_dim=attention_key_dim,
        dropout=attention_dropout,
        name="multi_head_self_attention",
    )(tokens, tokens)

    if use_residual:
        tokens = tf.keras.layers.Add(name="residual_attention_add")([tokens, attention_output])
    else:
        tokens = attention_output

    tokens = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="attention_layer_norm")(tokens)

    x = tf.keras.layers.GlobalAveragePooling1D(name="token_global_average_pooling")(tokens)
    x = tf.keras.layers.Dense(dense_units, activation="relu", name="dense_features")(x)
    x = tf.keras.layers.BatchNormalization(name="dense_batch_norm")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="classifier_dropout")(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="classification_head")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="EfficientNetB1_MHA_Residual")


def adaptive_boundary_loss(
    alpha: float = 0.15,
    margin: float = 0.20,
    label_smoothing: float = 0.0,
) -> Callable:
    """
    Adaptive boundary-aware categorical loss.

    The loss combines categorical cross-entropy with a confidence-margin penalty.
    The penalty encourages the predicted probability of the true class to remain
    sufficiently higher than the strongest competing class.

    Formula:
        L = CE(y, p) + alpha * max(0, margin - (p_true - p_competing))

    This formulation supports improved separation between visually similar
    classes while preserving the standard probabilistic interpretation of
    cross-entropy.
    """

    cross_entropy = tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=label_smoothing,
        reduction=tf.keras.losses.Reduction.NONE,
    )

    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())

        ce = cross_entropy(y_true, y_pred)

        true_class_probability = tf.reduce_sum(y_true * y_pred, axis=-1)

        competing_probabilities = (1.0 - y_true) * y_pred
        strongest_competing_probability = tf.reduce_max(competing_probabilities, axis=-1)

        class_margin = true_class_probability - strongest_competing_probability
        margin_penalty = tf.nn.relu(margin - class_margin)

        return ce + alpha * margin_penalty

    loss.__name__ = "adaptive_boundary_loss"
    return loss


def get_loss_function(config: Dict, model_name: str) -> Callable | str:
    """
    Return the loss function for a given model.

    The proposed model can use adaptive boundary-aware loss when enabled in
    config.yaml. Baseline models use categorical cross-entropy for fair reporting.
    """
    training_cfg = config.get("training", {})
    use_adaptive = bool(training_cfg.get("use_adaptive_loss_for_proposed", True))

    if model_name == "proposed" and use_adaptive:
        alpha = float(training_cfg.get("adaptive_loss_alpha", 0.15))
        margin = float(training_cfg.get("adaptive_loss_margin", 0.20))
        label_smoothing = float(training_cfg.get("label_smoothing", 0.0))
        return adaptive_boundary_loss(alpha=alpha, margin=margin, label_smoothing=label_smoothing)

    return "categorical_crossentropy"


def get_optimizer(config: Dict, model_name: str) -> tf.keras.optimizers.Optimizer:
    """Return optimizer configured for the selected model."""
    training_cfg = config.get("training", {})
    model_cfg = config.get("baseline_models", {}).get(model_name, {})

    optimizer_name = str(model_cfg.get("optimizer", training_cfg.get("optimizer", "adam"))).lower()
    learning_rate = float(model_cfg.get("learning_rate", training_cfg.get("learning_rate", 1e-4)))

    if optimizer_name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)

    if optimizer_name == "adamax":
        return tf.keras.optimizers.Adamax(learning_rate=learning_rate)

    if optimizer_name == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)

    if optimizer_name == "rmsprop":
        return tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

    raise ValueError(
        f"Unsupported optimizer '{optimizer_name}'. "
        "Use one of: adam, adamax, sgd, rmsprop."
    )


def build_model(model_name: str, config: Dict) -> tf.keras.Model:
    """Build a model by name."""
    model_name = model_name.lower().strip()

    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model '{model_name}'. "
            f"Supported models are: {sorted(SUPPORTED_MODELS)}"
        )

    if model_name == "cnn":
        return build_simple_cnn(config)

    if model_name == "vgg16":
        return build_vgg16(config)

    if model_name == "resnet50":
        return build_resnet50(config)

    if model_name == "efficientnetb1":
        return build_efficientnetb1_baseline(config)

    if model_name == "proposed":
        return build_proposed_efficientnetb1_mha_residual(config)

    raise RuntimeError("Unexpected model dispatch error.")


def compile_model(model: tf.keras.Model, model_name: str, config: Dict) -> tf.keras.Model:
    """Compile model with configured optimizer, loss, and metrics."""
    optimizer = get_optimizer(config, model_name)
    loss_function = get_loss_function(config, model_name)
    metrics = config.get("training", {}).get("metrics", ["accuracy"])

    model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=metrics,
    )
    return model


def build_and_compile_model(model_name: str, config: Dict) -> tf.keras.Model:
    """Convenience function to build and compile a model."""
    model = build_model(model_name, config)
    return compile_model(model, model_name, config)


def count_trainable_parameters(model: tf.keras.Model) -> int:
    """Count trainable parameters."""
    return int(
        sum(tf.keras.backend.count_params(weight) for weight in model.trainable_weights)
    )


def count_total_parameters(model: tf.keras.Model) -> int:
    """Count all parameters."""
    return int(model.count_params())


def estimate_flops_giga(model: tf.keras.Model) -> float:
    """
    Provide a lightweight approximate FLOPs estimate.

    Exact FLOPs calculation depends on TensorFlow graph tracing and runtime
    compatibility. This approximation is intentionally conservative and stable
    for reporting scripts. evaluate.py may replace this value with a graph-based
    estimate when available.
    """
    total_params = max(count_total_parameters(model), 1)

    # A rough inference-cost proxy: two operations per parameter plus attention
    # overhead for attention-enabled models. Returned as GFLOPs.
    multiplier = 2.0
    if "MHA" in model.name or "Attention" in model.name or "Residual" in model.name:
        multiplier = 2.15

    return round((total_params * multiplier) / 1e9, 4)


def get_model_summary_text(model: tf.keras.Model) -> str:
    """Return model summary as text."""
    lines = []
    model.summary(print_fn=lambda line: lines.append(line))
    return "\n".join(lines)


def save_model_summary(model: tf.keras.Model, output_path: str) -> None:
    """Save model summary to a text file."""
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(get_model_summary_text(model))


def get_custom_objects() -> Dict:
    """Return custom objects required when loading saved models."""
    return {
        "adaptive_boundary_loss": adaptive_boundary_loss(),
    }


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    import yaml

    parser = argparse.ArgumentParser(description="Build and inspect a model architecture.")
    parser.add_argument("--model", type=str, default="proposed", choices=sorted(SUPPORTED_MODELS))
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as config_file:
        cfg = yaml.safe_load(config_file)

    built_model = build_and_compile_model(args.model, cfg)
    built_model.summary()

    report_dir = Path(cfg.get("paths", {}).get("report_dir", "outputs/reports"))
    report_dir.mkdir(parents=True, exist_ok=True)

    summary_path = report_dir / f"{args.model}_model_summary.txt"
    save_model_summary(built_model, str(summary_path))

    print(f"Saved model summary to: {summary_path}")
    print(f"Total parameters: {count_total_parameters(built_model):,}")
    print(f"Trainable parameters: {count_trainable_parameters(built_model):,}")
    print(f"Approximate GFLOPs proxy: {estimate_flops_giga(built_model)}")
