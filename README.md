# 🧠 Neural Networks: From Scratch & TensorFlow / Keras Implementation

This repository contains the comprehensive implementation and study of **Artificial Neural Networks (ANN)**. It features an end-to-end modular neural network engine built completely from scratch using **NumPy**, as well as high-level deep learning implementations using **TensorFlow & Keras**.

---

## 📌 Table of Contents
- [Project Overview](#project-overview)
- [Mathematical Foundations](#mathematical-foundations)
- [Architecture Built From Scratch](#architecture-built-from-scratch)
- [TensorFlow / Keras Implementations](#tensorflow--keras-implementations)
- [Repository Structure](#repository-structure)
- [Requirements & Setup](#requirements--setup)
- [Results & Performance](#results--performance)
- [License](#license)

---

## 📖 Project Overview
The objective of this project is to demystify the internal mechanics of deep learning:
1. **Low-Level Implementation**: Developing vectorized Forward Propagation, Backpropagation, Activation derivatives, Loss functions, and Gradient Descent optimization using only Python and NumPy.
2. **High-Level Benchmarking**: Implementing dense multi-layer perceptrons (MLP) via TensorFlow/Keras to solve regression and computer vision classification tasks.

---

## 📐 Mathematical Foundations

### 1. Forward Propagation
For layer $l$ receiving input $A^{[l-1]}$:
$$Z^{[l]} = W^{[l]}A^{[l-1]} + B^{[l]}$$
$$A^{[l]} = f(Z^{[l]})$$
Where:
- $W^{[l]}$: Weight matrix
- $B^{[l]}$: Bias vector
- $f$: Activation function

### 2. Backpropagation & Parameter Updates
Given cost function $E$ and pre-activation gradients $D^{[l]}$:
$$D^{[l]} = D_u^{[l]} \odot f'(Z^{[l]})$$
$$\nabla_{W^{[l]}} E = D^{[l]} {A^{[l-1]}}^\mathsf{T}$$
$$\nabla_{B^{[l]}} E = D^{[l]}$$

Weight and bias updates with learning rate $\eta$:
$$W^{[l]} \leftarrow W^{[l]} - \eta \nabla_{W^{[l]}} E$$
$$B^{[l]} \leftarrow B^{[l]} - \eta \nabla_{B^{[l]}} E$$

---

## ⚙️ Architecture Built From Scratch (NumPy)

- **Activation Function**: `ReLU` (forward pass $\max(0, x)$ and backward derivative step)
- **Loss Function**: `MeanSquareError` (MSE loss calculation and derivative computation)
- **Optimizer**: `GradientDescent` (standard first-order gradient descent updates)
- **Layer Abstraction (`NeuronLayer`)**: Fully connected dense layer with uniform parameter initialization based on input/output dimensions
- **Model Pipeline (`Model`)**:
  - Mini-batch generation
  - Automated forward/backward multi-layer sequence traversal
  - `.compile()`, `.fit()`, `.predict()`, and `.evaluate()` APIs
  - Regression training on the **Boston Housing dataset**

---

## 🚀 TensorFlow / Keras Implementations

In addition to the scratch engine, a neural network pipeline is implemented using Keras:
- **Dataset**: MNIST Handwritten Digit Classification (60,000 training samples, 10,000 testing samples)
- **Data Normalization**: Scaling pixel values from $[0, 255]$ to $[0.0, 1.0]$
- **Model Architecture**:
  - `Flatten(input_shape=(28, 28))`
  - `Dense(128, activation='relu')`
  - `Dense(10, activation='softmax')`
- **Loss & Optimizer**: `SparseCategoricalCrossentropy` with `Adam(lr=0.001)`

---

## 📁 Repository Structure

```text
├── notebooks/
│   └── cse422_neural_networks.ipynb   # Complete lab notebook with scratch & Keras code
├── src/
│   ├── activations.py                 # Custom activation functions (ReLU, etc.)
│   ├── losses.py                      # Loss functions (MSE, Cross-Entropy)
│   ├── optimizers.py                  # Gradient Descent optimizers
│   └── neural_network.py              # Layer and Model class implementations
├── requirements.txt                   # Environment dependencies
├── .gitignore
└── README.md                          # Project documentation
