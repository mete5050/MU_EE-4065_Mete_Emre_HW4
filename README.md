
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
