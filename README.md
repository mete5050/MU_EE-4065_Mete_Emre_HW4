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
