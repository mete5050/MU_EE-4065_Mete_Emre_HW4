## Q1 (Book: Section 10.9) — Single Neuron Binary Classifier (0 vs not-0)

### Objective
In this task, we classify MNIST digits into two classes:
- Class 0: digit "0"
- Class 1: "not-0" (all digits 1–9)

This is a binary classification problem.  
> Important: In the Q1 confusion matrix, the labels 0 and 1 do not mean digit "0" and digit "1".  
They mean binary classes: 0 = zero, 1 = not-zero.

---

### Dataset (Offline MNIST) — Train/Test Inputs
We use the offline MNIST IDX files:

Train split
- MNIST-dataset/train-images.idx3-ubyte → 60,000 grayscale images, each 28×28
- MNIST-dataset/train-labels.idx1-ubyte → 60,000 labels (0–9)

Test split
- MNIST-dataset/t10k-images.idx3-ubyte → 10,000 grayscale images, each 28×28
- MNIST-dataset/t10k-labels.idx1-ubyte → 10,000 labels (0–9)

How labels are prepared for Q1 (binary mapping):
- If label == 0 → mapped label = 0
- If label != 0 → mapped label = 1

---

### Algorithm / Pipeline
The original MNIST images are not directly fed to the classifier. Instead:

1) Feature Extraction (Hu Moments)
- For each 28×28 image, we compute image moments and then extract 7 Hu Moments.
- This converts each image into a 7-dimensional feature vector.
- Motivation: Hu Moments provide compact shape descriptors, reducing input size drastically (784 pixels → 7 features), which fits the embedded ML perspective.

2) Feature Normalization
- We compute mean and standard deviation on the training features and standardize both train and test features:
  - X_norm = (X - mean) / std
- This helps the model train more stably.

3) Model: Single Neuron (Logistic Regression style)
- Architecture: Dense(1, activation="sigmoid")
- Output is a probability in [0, 1] for the positive class (not-0).

4) Training
- Loss: Binary Cross Entropy
- Optimizer: Adam
- Class weighting: {0: 8, 1: 1} (digit “0” is less frequent than not-0, so this balances the loss contribution)

---

### Results (Test Set)
Artifacts produced
- Confusion matrix image: output/q1_confusion_matrix.png
- Saved model: output/mnist_single_neuron.h5

Accuracy calculation
From your confusion matrix:
- Correct predictions = 945 (true 0 predicted 0) + 7511 (true 1 predicted 1) = 8456
- Total test samples = 10,000
- Accuracy = 8456 / 10000 = 0.8456 = 84.56%

Interpretation
- The model performs well for a simple binary task with compact features.
- The largest error comes from not-0 samples predicted as 0 (1509 cases), meaning some non-zero digits have Hu-Moment patterns close to zeros.

---

## Q2 (Book: Section 11.8) — Multi-Layer Perceptron (MLP) (0–9)

### Objective
In this task, we classify MNIST digits into 10 classes:
- Classes are the actual digits: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9

This is a multi-class classification problem.

---

### Dataset (Offline MNIST) — Train/Test Inputs
We use the same offline MNIST IDX files:

Train split
- MNIST-dataset/train-images.idx3-ubyte → 60,000 images (28×28)
- MNIST-dataset/train-labels.idx1-ubyte → 60,000 labels (0–9)

Test split
- MNIST-dataset/t10k-images.idx3-ubyte → 10,000 images (28×28)
- MNIST-dataset/t10k-labels.idx1-ubyte → 10,000 labels (0–9)

Unlike Q1, labels are not converted. We keep the 0–9 labels as they are.

---

### Algorithm / Pipeline
Again, we do not feed raw pixels to the network. We follow the same feature pipeline:

1) Feature Extraction (Hu Moments)
- Each image → 7 Hu Moments → a 7D feature vector

2) Model: MLP (Neural Network)
Architecture:
- Dense(100, ReLU)
- Dense(100, ReLU)
- Dense(10, Softmax)

The softmax layer outputs probabilities for the 10 digit classes.

3) Training
- Loss: Sparse Categorical Cross Entropy
- Optimizer: Adam (learning rate = 1e-4)
- Callbacks:
  - EarlyStopping (stops if training loss stops improving)
  - ModelCheckpoint (saves best model)

---

### Results (Test Set)
Artifacts produced
- Confusion matrix image: output/q2_confusion_matrix.png
- Saved model: output/mlp_mnist_model.h5

Accuracy calculation
From the confusion matrix, the diagonal sum (correct predictions) is:
- 5852 correct out of 10,000
- Accuracy = 5852 / 10000 = 0.5852 = 58.52%

  Interpretation
- This accuracy is moderate for 10-class MNIST, and it is expected because:
  - The input is only 7 features (Hu Moments), not the full pixel grid.
  - Many digits have similar global shapes (e.g., 3 vs 5, 4 vs 9, etc.), which can be difficult to separate using only Hu Moments.
- The confusion matrix shows which digits are mixed with others (off-diagonal values), indicating where the feature representation is insufficient.

---

### How to Read the Confusion Matrices (applies to both)
- Rows (Y-axis) = True label (ground-truth)
- Columns (X-axis) = Predicted label (model output)
- Diagonal entries = correct predictions
- Off-diagonal entries = misclassifications

For Q1:
- 0 and 1 are binary classes, not digits 0 and 1.
For Q2:
- 0–9 are the actual digit classes.
