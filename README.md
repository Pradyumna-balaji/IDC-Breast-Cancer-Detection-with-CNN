# 🧠 IDC Breast Cancer Detection with CNN

This project focuses on classifying histopathological images to detect **Invasive Ductal Carcinoma (IDC)** — the most common form of breast cancer — using **Convolutional Neural Networks (CNNs)** with **TensorFlow/Keras**.

---

## 📂 Dataset

The dataset used is the **IDC Regular Histopathological Cancer Dataset** available from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images).

- Each image is **50x50** patches of breast tissue.
- Two classes:
  - `0` - No IDC (non-cancerous)
  - `1` - IDC present (cancerous)

---

## 🚀 Features

- Custom CNN architecture using `SeparableConv2D` layers for efficient learning.
- Image augmentation using `ImageDataGenerator` to improve generalization.
- Full training pipeline: preprocessing, training, evaluation, and visualization.
- Metrics: Accuracy, Sensitivity, Specificity, Confusion Matrix.
- Results visualized via loss and accuracy plots.
- Works in **Jupyter Notebooks** and standard Python scripts.

---

## 🛠️ Technologies

- Python 3
- TensorFlow & Keras
- NumPy, Matplotlib, Scikit-learn
- imutils
- Jupyter Notebook

---

## 🧪 Training and Evaluation

After preparing the dataset and running the training pipeline, the model:

- Evaluates on a separate test set.
- Generates classification report and confusion matrix.
- Computes:
  - **Accuracy**
  - **Sensitivity (Recall for IDC class)**
  - **Specificity (Recall for Non-IDC class)**

---

## 📊 Visualization

Training accuracy and loss are plotted and saved as:

