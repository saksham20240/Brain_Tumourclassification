# Brain_Tumourclassification

A deep learning pipeline that classifies brain MRI scans into four categories
(glioma, meningioma, pituitary, no tumor) and explains its predictions using
Grad-CAM. Built and tuned to run locally on a 4GB consumer GPU.

*Python · TensorFlow/Keras · EfficientNetB1 · OpenCV · scikit-learn*

## Overview

The dataset ships pre-labeled into the four classes above. This project
handles everything from there: cleaning up each scan with OpenCV-based
cropping, training a transfer-learned EfficientNetB1 classifier on the
result, evaluating it properly on a held-out test set, and generating
Grad-CAM heatmaps so a prediction can be inspected rather than trusted
blindly.

## Results

| Metric | Score |
|---|---|
| Test accuracy | 95.1% |
| Weighted F1 | 0.95 |

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Glioma | 0.99 | 0.84 | 0.91 |
| Meningioma | 0.91 | 0.97 | 0.94 |
| No tumor | 0.94 | 1.00 | 0.97 |
| Pituitary | 0.96 | 0.99 | 0.98 |

Glioma recall is the weak point — about 1 in 6 true glioma scans get
misclassified as something else. Precision on that class is still high
(0.99), so the model rarely mislabels other classes *as* glioma; it's
specifically under-catching gliomas. Likely candidates for improving this:
class-weighted loss, glioma-specific augmentation, or auditing the
contour-cropping step for tumors near the edge of the frame.

## Pipeline

1. **Preprocess** — OpenCV contour detection to crop each scan to the brain
   region, then resize and cache to disk.
2. **Train** — EfficientNetB1 (ImageNet weights) with a global-max-pool and
   dropout head, trained with mixed precision and GPU memory growth so it
   fits in 4GB of VRAM.
3. **Evaluate** — confusion matrix and full classification report on the
   test set.
4. **Predict** — batched inference with a sample-grid visualization.
5. **Grad-CAM** — gradient-based heatmap showing which part of a scan drove
   a given prediction.

Each stage is runnable independently once the previous one's output exists.

## Setup

```bash
conda env create -f environment.yml
conda activate brain-tumor-gpu
```

TensorFlow's `[and-cuda]` pip extra pulls in matching CUDA/cuDNN libraries,
so no separate system-wide CUDA install is required (an NVIDIA driver still
is).

## Dataset

[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
(Kaggle). Download and place it as:

```
dataset/
├── Training/{glioma,meningioma,notumor,pituitary}/
└── Testing/{glioma,meningioma,notumor,pituitary}/
```

## Usage

```bash
python brain_tumor_pipeline.py --stage all       # run everything
python brain_tumor_pipeline.py --stage train      # just (re)train
python brain_tumor_pipeline.py --stage predict    # just run inference
python brain_tumor_pipeline.py --stage gradcam    # just generate Grad-CAM
```

`--batch-size` (default 8, tuned for 4GB VRAM), `--epochs` (default 30), and
`--force-preprocess` (re-crop even if cached output exists) are also
available.

## Project structure

```
.
├── brain_tumor_pipeline.py
├── environment_minimal.yml
├── training_history.png
├── confusion_matrix.png
├── sample_predictions.png
├── gradcam_example.png
└── README.md
```

`dataset/`, `Crop-Brain-MRI/`, `Test-Data/`, and `model.keras` are
regenerated locally rather than committed.
