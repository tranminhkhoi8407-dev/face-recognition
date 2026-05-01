# Face Recognition with PCA (Eigenfaces)

Face recognition system built **from scratch** using **Principal Component Analysis (PCA)** —
also known as the **Eigenfaces** method — on the AT&T (ORL) face dataset.
No pre-built ML or face recognition libraries are used.
The entire PCA pipeline is implemented manually with NumPy.

---

## Overview

This project demonstrates how linear algebra concepts — covariance matrices, eigenvalues,
and eigenvectors — can be applied to solve a real-world computer vision problem.
Each face image is compressed from a 10,304-dimensional vector down to k dimensions
using the top-k eigenfaces, then recognized via nearest-neighbor search in PCA space.

---

## Dataset

**AT&T Face Database (ORL)**
- 40 subjects × 10 grayscale images = **400 images total**
- Image size: **112 × 92 pixels** → flattened to a 10,304-dim vector
- Split: first 7 images/subject → **train (280 images)**, last 3 → **test (120 images)**

> Place the dataset folder in the same directory as `pca.py` and rename it `ATnT/`.
> Expected structure: `ATnT/s1/1.pgm`, `ATnT/s1/2.pgm`, ..., `ATnT/s40/10.pgm`

---

## Pipeline

| Step | Function | Description |
|------|----------|-------------|
| 1 | `load_data()` | Read `.pgm` images, flatten each 112×92 matrix to a 10,304-dim float64 vector, split into train/test |
| 2 | `compute_pca()` | Compute mean face, center data, apply the **dual trick** (eigen-decompose A·Aᵀ of size 280×280 instead of Aᵀ·A of size 10304×10304), project back to image space, sort by descending eigenvalue |
| 3 | `extract_features()` | Keep top-k eigenfaces; project all training images → weight matrix of shape (280, k) |
| 4 | `recognize_face()` | Normalize test image, project into PCA space, find nearest neighbor by Euclidean distance |
| 5 | `evaluate_accuracy()` | Sweep over multiple k values, compute recognition accuracy, plot and save the Accuracy vs k curve |
| Bonus | `show_eigenfaces()` | Visualize the mean face and first 10 eigenfaces |

### Data flow
400 images
→ flatten → (280 × 10304) train matrix
→ PCA     → 280 eigenfaces
→ keep k  → (280 × k) weight matrix
→ 1-NN    → predicted label
---

## Results

| k (components) | Accuracy |
|:--------------:|:--------:|
| 5              | ~80%     |
| 20             | ~93%     |
| 50             | ~95%+    |
| 80–150         | ~96%+    |

> Best accuracy is typically achieved around **k = 80–150**.
> See `accuracy_vs_k.png` for the full curve.

---

## Outputs

| File | Description |
|------|-------------|
| `eigenfaces.png` | Mean face + first 10 eigenfaces visualized as images |
| `accuracy_vs_k.png` | Recognition accuracy plotted against number of components k |

---

## Getting Started

### 1. Install dependencies

```bash
pip install numpy opencv-python matplotlib
```

### 2. Prepare the dataset
```plaintext
project/
├── pca.py
└── ATnT/
├── s1/
│   ├── 1.pgm
│   └── ...
├── s2/
└── ...
### 3. Run

```bash
python pca.py
```

The program will:
1. Load and split the dataset
2. Compute PCA and display eigenfaces (close the popup to continue)
3. Evaluate recognition accuracy for k ∈ {5, 10, 20, 30, 50, 80, 100, 150, 200}
4. Print the best k and accuracy, save both plots

---

## Requirements
1. numpy

2. opencv-python

3. matplotlib
## Project Structure

```plaintext
project/
├── pca.py                # Main script: PCA + Eigenfaces pipeline
├── requirements.txt      # Dependencies (numpy, opencv-python, matplotlib)
├── README.md             # Project documentation
│
├── ATnT/                 # Dataset folder (40 subjects × 10 images each)
│   ├── s1/
│   │   ├── 1.pgm
│   │   ├── 2.pgm
│   │   └── ...
│   ├── s2/
│   ├── ...
│   └── s40/
│
├── outputs/              # Generated results
│   ├── eigenfaces.png    # Mean face + first 10 eigenfaces
│   └── accuracy_vs_k.png # Accuracy vs number of components k

