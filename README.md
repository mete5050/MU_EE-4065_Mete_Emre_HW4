
# EE4065 — Homework 4 (Embedded Machine Learning)

This repository contains implementations of two end-of-chapter applications from the book Embedded Machine Learning with Microcontrollers:

- Q1 (Section 10.9): Single Neuron (binary) classifier  
- Q2 (Section 11.8): Multi-Layer Perceptron (MLP) classifier

Both tasks use the offline MNIST dataset (train/test IDX files). Each MNIST image is converted into a compact 7-dimensional feature vector using Hu Moments (feature extraction), which is suitable for embedded/low-resource ML pipelines.

---

## Repository Structure

.
├─ MNIST-dataset/
│  ├─ train-images.idx3-ubyte
│  ├─ train-labels.idx1-ubyte
│  ├─ t10k-images.idx3-ubyte
│  └─ t10k-labels.idx1-ubyte
├─ output/
│  ├─ q1_confusion_matrix.png
│  ├─ q2_confusion_matrix.png
│  ├─ mnist_single_neuron.h5
│  └─ mlp_mnist_model.h5
├─ mnist.py
├─ q1_mnist_single_neuron.py
└─ q2_mlp_mnist.py

> 
---

## Dataset (Offline MNIST) — Train/Test

MNIST is provided as two splits:

### Train set
- train-images.idx3-ubyte — 60,000 images, 28×28 grayscale
- train-labels.idx1-ubyte — 60,000 labels (0–9)

### Test set
- t10k-images.idx3-ubyte — 10,000 images, 28×28 grayscale
- t10k-labels.idx1-ubyte — 10,000 labels (0–9)

How labels are used in each question:
- Q1 (binary): labels are converted to two classes  
  - 0 = digit "0"  
  - 1 = "not-0" (all digits 1–9)
- Q2 (10-class): labels remain 0–9

---

## Environment & Dependencies

Recommended: Python 3.9+

Install required packages:

pip install numpy opencv-python scikit-learn matplotlib tensorflow

---

## How to Run

### Q1 — Single Neuron (0 vs not-0)

python q1_mnist_single_neuron.py

Outputs:

* output/q1_confusion_matrix.png
* output/mnist_single_neuron.h5

### Q2 — MLP (0–9)

python q2_mlp_mnist.py

Outputs:

* output/q2_confusion_matrix.png
* output/mlp_mnist_model.h5

> mnist.py is a helper module for reading IDX files; you do not run it directly.

---

## Results

### Q1 — Confusion Matrix (Binary Classification)

This confusion matrix uses:

* True label on the Y-axis
* Predicted label on the X-axis
* Diagonal values are correct predictions.

![Q1 Confusion Matrix](output/q1_confusion_matrix.png)

Q1 Accuracy:

* Correct = 945 + 7511 = 8456
* Total test samples = 10,000
* Accuracy = 8456 / 10000 = 0.8456 (84.56%)

Short comment: The model performs reasonably well for separating digit 0 vs not-0, but it still confuses some not-0 samples as 0.

---

### Q2 — Confusion Matrix (10-Class Classification)

This confusion matrix uses:

* True label (0–9) on the Y-axis
* Predicted label (0–9) on the X-axis
* Diagonal values are correct predictions.

![Q2 Confusion Matrix](output/q2_confusion_matrix.png)

Q2 Accuracy:

* Diagonal (correct) sum = 5852
* Total test samples = 10,000
* Accuracy = 5852 / 10000 = 0.5852 (58.52%)

Short comment: Multi-class classification is harder with only 7 Hu Moments as input features, so the model mixes visually similar digits more often.

---

## Code Snippets (Required in the Report)

### 1) IDX File Reader (mnist.py)

Core idea used to read MNIST IDX files:

import struct
import numpy as np

def load_images(path):
    with open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
    return data

def load_labels(path):
    with open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num)
    return data

### 2) Feature Extraction (Hu Moments)

Each image is converted into a 7D feature vector:

import cv2
import numpy as np

hu = np.empty((len(images), 7), dtype=np.float32)
for i, img in enumerate(images):
    m = cv2.moments(img, True)
    hu[i] = cv2.HuMoments(m).reshape(7)

### 3) Model Definitions

Q1 (single neuron / sigmoid):

import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=[7], activation="sigmoid")
])

Q2 (MLP / softmax):

from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(100, input_shape=[7], activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

Full implementations are available in:

* q1_mnist_single_neuron.py
* q2_mlp_mnist.py

---

## Notes on Reading the Confusion Matrices

* Rows (Y-axis) show the true classes.
* Columns (X-axis) show the predicted classes.
* Values on the diagonal are correct predictions.
* Off-diagonal values indicate misclassifications.

For Q1, the labels 0 and 1 are not digits "0" and "1":

* 0 means digit 0
* 1 means not-0 (digits 1–9)

---
