#!/usr/bin/env bash
set -euo pipefail

# Reproducibility runner for:
# EfficientNetB1-MHA-Residual-GI-Classification
#
# This script trains and evaluates the proposed model and all compared
# baseline models using the same configuration, preprocessing pipeline,
# augmentation policy, and evaluation protocol.

CONFIG_FILE="${1:-config.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"

MODELS=(
  "cnn"
  "vgg16"
  "resnet50"
  "efficientnetb1"
  "proposed"
)

echo "============================================================"
echo "EfficientNetB1-MHA-Residual-GI-Classification"
echo "Full Reproducibility Run"
echo "============================================================"
echo "Configuration file: ${CONFIG_FILE}"
echo "Python executable : ${PYTHON_BIN}"
echo "Started at        : $(date)"
echo "============================================================"

if [ ! -f "${CONFIG_FILE}" ]; then
  echo "ERROR: Configuration file not found: ${CONFIG_FILE}"
  echo "Run this script from the repository root or pass the config path:"
  echo "bash run_reproducibility.sh config.yaml"
  exit 1
fi

if [ ! -d "src" ]; then
  echo "ERROR: src/ directory not found."
  echo "Run this script from the repository root."
  exit 1
fi

if [ ! -d "data/train" ] || [ ! -d "data/val" ] || [ ! -d "data/test" ]; then
  echo "ERROR: Dataset folders are missing."
  echo "Expected dataset structure:"
  echo "data/train/<class_name>/"
  echo "data/val/<class_name>/"
  echo "data/test/<class_name>/"
  exit 1
fi

mkdir -p outputs/checkpoints outputs/logs outputs/reports outputs/figures outputs/environment

echo ""
echo "Saving environment summary..."
{
  echo "Run date: $(date)"
  echo "Python executable: ${PYTHON_BIN}"
  echo ""
  ${PYTHON_BIN} --version
  echo ""
  ${PYTHON_BIN} -m pip freeze
} > outputs/environment/environment_summary.txt

echo ""
echo "Dataset folder summary:"
find data -maxdepth 2 -type d | sort

echo ""
echo "============================================================"
echo "Training and evaluating compared models"
echo "============================================================"

for MODEL_NAME in "${MODELS[@]}"; do
  echo ""
  echo "------------------------------------------------------------"
  echo "Model: ${MODEL_NAME}"
  echo "------------------------------------------------------------"

  echo "Training ${MODEL_NAME}..."
  ${PYTHON_BIN} src/train.py     --model "${MODEL_NAME}"     --config "${CONFIG_FILE}"

  echo "Evaluating ${MODEL_NAME}..."
  ${PYTHON_BIN} src/evaluate.py     --model "${MODEL_NAME}"     --config "${CONFIG_FILE}"
done

echo ""
echo "============================================================"
echo "Generating final plots and comparison summaries"
echo "============================================================"
${PYTHON_BIN} src/plots.py --config "${CONFIG_FILE}"

echo ""
echo "============================================================"
echo "Reproducibility run completed successfully"
echo "Completed at: $(date)"
echo "Generated outputs are available in the outputs/ directory."
echo "============================================================"
