# Human Activity Recognition (UCI-HAR) using ANN

## Project Overview

This project implements a **Human Activity Recognition (HAR)** system using the **UCI HAR Dataset** and a **tuned Artificial Neural Network (ANN)**.  
The goal is to classify human activities based on smartphone sensor data using **engineered features** and evaluate the model on an unseen test set.

A fully tuned ANN model achieves a **test accuracy of 96.02%**, demonstrating strong generalization performance.

---

## Dataset Description

The **UCI HAR Dataset** contains data collected from smartphone sensors (accelerometer and gyroscope) worn by participants while performing daily activities.

### Activities (Target Classes)

| Label | Activity |
|-----|---------|
| 1 | WALKING |
| 2 | WALKING_UPSTAIRS |
| 3 | WALKING_DOWNSTAIRS |
| 4 | SITTING |
| 5 | STANDING |
| 6 | LAYING |

---

## Data Representation

This project uses **engineered features**, not raw time-series signals.

### Feature File Used
- `X_train.txt`
- `X_test.txt`

Each sample contains **561 hand-engineered features**, including:
- Time-domain statistics (mean, std, energy)
- Frequency-domain features (FFT-based)
- Signal magnitude and correlation features

These features are **pre-normalized** in the dataset.

### Label Files
- `y_train.txt`
- `y_test.txt`

Labels are integer encoded and shifted to **0–5** for model compatibility.

---

## Why ANN (and not LSTM/CNN)?

- Features are already **domain-engineered**
- Temporal patterns are already summarized
- ANN performs extremely well on this representation
- No need for sequence modeling

> Raw inertial signals are suitable for CNN/LSTM, but engineered features are best handled by ANN or classical ML models.

---

## Model Architecture

A fully connected ANN with:
- Tunable depth (3–6 hidden layers)
- Tunable width (16–256 neurons per layer)
- ReLU / PReLU / Tanh activations
- Batch Normalization
- Dropout regularization

Final layer:
- Softmax activation (6 classes)

Loss function:
- `sparse_categorical_crossentropy`

Optimizer:
- Adam / RMSprop (tuned)

---

## Hyperparameter Tuning

Hyperparameters were optimized using **Keras Tuner (Random Search)**.

### Tuned Parameters
- Number of hidden layers
- Units per layer
- Activation functions
- Dropout rates
- Optimizer type
- Learning rate

### Regularization
- Batch Normalization
- Dropout
- Early Stopping (validation loss)

---

## Training Strategy

- Validation split: 20%
- Early stopping with `restore_best_weights=True`
- Final model retrained from scratch using best hyperparameters

---

## Model Performance

### Test Set Results

- **Test Accuracy:** **96.02%**

Additional evaluation includes:
- Confusion matrix
- Precision, recall, F1-score
- ROC-AUC (for robustness)

The model demonstrates strong generalization and minimal overfitting.

---

## Key Takeaways

- Hand-engineered HAR features are highly discriminative
- ANN performs exceptionally well without PCA
- PCA was avoided to preserve semantic feature information
- Hyperparameter tuning significantly improved performance
- Early stopping ensured stable generalization

---

## Technologies Used

- Python
- TensorFlow / Keras
- Keras Tuner
- NumPy
- Scikit-learn
- Matplotlib

---

## Project Structure
