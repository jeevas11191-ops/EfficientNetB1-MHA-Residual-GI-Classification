# EfficientNetB1-MHA-Residual-GI-Classification

This repository provides the reproducible implementation of an endoscopy image classification framework using **EfficientNetB1 with Multi-Head Attention and Residual Connections**. The framework is designed for four-class gastrointestinal image classification involving:

- Normal
- Ulcerative Colitis
- Polyps
- Esophagitis

The implementation includes preprocessing, augmentation, baseline model comparison, proposed model training, evaluation, confusion matrix generation, ROC analysis, training curves, computational summary, and ablation-oriented reporting.

---

## 1. Project Overview

Automated gastrointestinal image classification can support faster screening and decision-making in endoscopy-based diagnosis. Manual interpretation of endoscopic and capsule endoscopy images is time-consuming and may be affected by observer variability. This project implements a compact deep learning framework that combines the feature efficiency of EfficientNetB1 with attention-guided spatial refinement and residual feature preservation.

The proposed model follows the pipeline:

```text
Input Image
→ EfficientNetB1 Backbone
→ Final Convolutional Feature Map
→ Spatial Token Reshaping
→ Multi-Head Self-Attention
→ Residual Feature Addition
→ Global Average Pooling
→ Dropout
→ Softmax Classification
```

The repository also includes baseline models for comparative evaluation under the same dataset split, preprocessing pipeline, and training configuration.

---

## 2. Repository Structure

```text
EfficientNetB1-MHA-Residual-GI-Classification/
│
├── README.md
├── requirements.txt
├── config.yaml
├── run_reproducibility.sh
├── .gitignore
│
└── src/
    ├── data_pipeline.py
    ├── models.py
    ├── train.py
    ├── evaluate.py
    └── plots.py
```

---

## 3. Dataset Preparation

The dataset should be arranged manually before running the code. The implementation expects the following folder structure:

```text
data/
├── train/
│   ├── normal/
│   ├── ulcerative_colitis/
│   ├── polyps/
│   └── esophagitis/
│
├── val/
│   ├── normal/
│   ├── ulcerative_colitis/
│   ├── polyps/
│   └── esophagitis/
│
└── test/
    ├── normal/
    ├── ulcerative_colitis/
    ├── polyps/
    └── esophagitis/
```

Recommended dataset distribution:

| Class | Train | Validation | Test |
|---|---:|---:|---:|
| Normal | 800 | 500 | 200 |
| Ulcerative Colitis | 800 | 500 | 200 |
| Polyps | 800 | 500 | 200 |
| Esophagitis | 800 | 500 | 200 |

Total images:

| Split | Images |
|---|---:|
| Train | 3200 |
| Validation | 2000 |
| Test | 800 |
| Total | 6000 |

The dataset can be prepared from publicly available gastrointestinal endoscopy image datasets such as KVASIR and ETIS-Larib Polyp DB. Dataset files are not included in this repository.

---

## 4. Installation

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

For Windows:

```bash
venv\Scripts\activate
```

For Linux or macOS:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 5. Preprocessing and Augmentation

All images are processed using a unified preprocessing pipeline.

### Preprocessing

| Step | Description |
|---|---|
| RGB conversion | Ensures all input images have three channels |
| Resizing | Images are resized to 224 × 224 |
| Normalization | Pixel values are scaled to [0, 1] using 1/255 normalization |
| Split control | Training, validation, and testing folders are loaded independently |

### Training-Time Augmentation

Augmentation is applied only to the training set.

| Augmentation | Value |
|---|---:|
| Rotation range | 49 |
| Height shift | 0.1 |
| Width shift | 0.2 |
| Brightness range | 0.1 to 3.0 |
| Horizontal flip | True |
| Vertical flip | True |
| Shear range | 0.2 |
| Zoom range | 0.3 |
| Gaussian noise | Enabled |

These transformations improve generalization and reduce overfitting under limited-data conditions.

---

## 6. Compared Models

The repository supports training and evaluation of the following models:

| Model Name | Description |
|---|---|
| `cnn` | Lightweight convolutional baseline |
| `vgg16` | Transfer learning baseline using VGG16 |
| `resnet50` | Transfer learning baseline using ResNet50 |
| `efficientnetb1` | EfficientNetB1 baseline without attention and residual refinement |
| `proposed` | EfficientNetB1 with Multi-Head Attention and Residual Connections |

---

## 7. Hyperparameter Configuration

All major hyperparameters are stored in `config.yaml`.

Default configuration:

| Hyperparameter | Value |
|---|---:|
| Image size | 224 × 224 |
| Batch size | 16 |
| Epochs | 100 |
| Optimizer | Adam |
| Initial learning rate | 0.0001 |
| Loss function | Categorical cross-entropy / adaptive loss option |
| Dropout | 0.5 |
| Attention heads | 4 |
| Attention key dimension | 64 |
| Classes | 4 |
| Random seed | 42 |

Callbacks used during training:

| Callback | Purpose |
|---|---|
| ModelCheckpoint | Saves best validation model |
| EarlyStopping | Stops training if validation loss does not improve |
| ReduceLROnPlateau | Reduces learning rate during stagnation |
| CSVLogger | Saves epoch-wise training history |

---

## 8. Training

Train the proposed model:

```bash
python src/train.py --model proposed --config config.yaml
```

Train EfficientNetB1 baseline:

```bash
python src/train.py --model efficientnetb1 --config config.yaml
```

Train ResNet50 baseline:

```bash
python src/train.py --model resnet50 --config config.yaml
```

Train VGG16 baseline:

```bash
python src/train.py --model vgg16 --config config.yaml
```

Train CNN baseline:

```bash
python src/train.py --model cnn --config config.yaml
```

---

## 9. Evaluation

Evaluate the proposed model:

```bash
python src/evaluate.py --model proposed --config config.yaml
```

Evaluate a baseline model:

```bash
python src/evaluate.py --model efficientnetb1 --config config.yaml
```

The evaluation script generates:

```text
outputs/
├── reports/
│   ├── proposed_classification_report.csv
│   ├── proposed_metrics.json
│   └── proposed_confusion_matrix.csv
│
├── figures/
│   ├── proposed_confusion_matrix.png
│   ├── proposed_roc_curve.png
│   ├── proposed_accuracy_curve.png
│   └── proposed_loss_curve.png
│
└── logs/
    └── proposed_training_log.csv
```

---

## 10. Full Reproducibility Run

To run the main reproducibility pipeline:

```bash
bash run_reproducibility.sh
```

This command trains and evaluates the proposed model and selected baseline models using the same dataset split, preprocessing settings, augmentation strategy, and hyperparameter configuration.

---

## 11. Output Files

After training and evaluation, the following outputs are generated:

| Output | Description |
|---|---|
| Training logs | Epoch-wise accuracy and loss |
| Classification report | Precision, recall, F1-score, and support |
| Confusion matrix | Class-wise prediction distribution |
| ROC curve | Multi-class ROC analysis |
| Accuracy curve | Training and validation accuracy |
| Loss curve | Training and validation loss |
| Metrics file | Accuracy, macro precision, macro recall, macro F1-score, inference time |
| Hyperparameter summary | Training settings used for each model |

---

## 12. Model Architecture Summary

The proposed architecture uses EfficientNetB1 as the feature extraction backbone. The classification head of EfficientNetB1 is removed, and the final convolutional feature map is reshaped into spatial tokens. Multi-head self-attention is then applied to learn long-range feature dependencies across spatial regions. A residual connection combines the original EfficientNetB1 feature tokens with the attention-refined feature tokens. The resulting representation is aggregated using global average pooling, regularized using dropout, and classified using a dense softmax layer.

Architecture flow:

```text
Input: 224 × 224 × 3
↓
EfficientNetB1 backbone without classifier
↓
Feature map: 7 × 7 × 1280
↓
Reshape: 49 × 1280
↓
Multi-Head Self-Attention
↓
Residual Addition
↓
Global Average Pooling
↓
Dropout
↓
Dense Softmax Layer
↓
Output: 4 classes
```

---

## 13. Experimental Reporting

The implementation supports direct reporting of:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- ROC-AUC
- Inference time
- Trainable parameter count
- Approximate computational complexity
- Training and validation convergence

These outputs are intended to support transparent comparison between the proposed model and baseline methods.

---

## 14. Reproducibility Statement

This repository provides the implementation details, preprocessing pipeline, augmentation settings, model architectures, training configuration, evaluation scripts, and visualization utilities required to reproduce the gastrointestinal image classification experiments. All compared models are trained using the same dataset structure, image resolution, optimization settings, and evaluation protocol to ensure a fair comparison.

The dataset must be downloaded and organized separately by the user because medical image datasets may have independent access, citation, and usage requirements.

---

## 15. Notes

- The repository does not include dataset files.
- The repository does not include trained model weights by default.
- The repository does not include any license file.
- The repository does not include DOI information.
- All generated results are saved automatically inside the `outputs/` directory.
