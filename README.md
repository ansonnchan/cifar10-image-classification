# CIFAR-10 Image Classification with PyTorch

**A modular deep learning pipeline for identifying objects in the CIFAR-10 dataset using Convolutional Neural Networks (CNNs).**

## Project Overview
This project implements an end-to-end machine learning workflow to classify images from the CIFAR-10 dataset into 10 distinct categories. The pipeline is designed with modularity in mind, separating data orchestration, model architecture, training loops, and evaluation logic.

### Key Features:
* **Modular Architecture:** Easily swap models or datasets by modifying specific source files.
* **Automated Pipeline:** Handles data augmentation, training, and multi-metric evaluation.
* **Error Analysis:** Generates confusion matrices and visualizes specific images the model failed to classify correctly.
* **Best-Model Saving:** Automatically checkpoints the model with the highest validation accuracy.

---

## Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/ansonnchan/cifar10-image-classification
cd cifar10-image-classification
```

### 2. Check Requirements
This project requires libraries such as `torch`, `torchvision`, `seaborn`, and `sklearn` (more in **requirements.txt**). 

You can download all of them by running the command below:
```bash
pip install -r requirements.txt
```

### 3. Run the Project
```bash
python main.py
```
**Note on Data:** The CIFAR-10 dataset is large (~160MB). You do not need to download it manually; the script will automatically download and extract it for you upon the first run. 


## Training Reflection & Results

### Performance Summary
The model was trained for **20 epochs**, taking approximately **22 minutes** in total. On the hardware used, each epoch required roughly **1 minute** of processing time.

* **Training Dynamics:** We observed a standard learning curve where the model gained significant accuracy in the first 10 epochs. Toward the end of the 20-epoch run, improvements in loss and accuracy became minimal, indicating the model was approaching its convergence point.
* **Final Performance (Epoch 20):**
    * **Training Loss:** `0.6582` | **Training Accuracy:** `77.39%`
    * **Validation Loss:** `0.6726` | **Validation Accuracy:** `76.52%`

![Alt Text](https://github.com/ansonnchan/cifar10-image-classification/blob/main/imgs/cifar_vis.png)

### Detailed Classification Report
The model achieved a global accuracy of **76%**. The report below highlights the model's strengths and weaknesses across the 10 categories. Notably, it excels at identifying **Automobiles** (96% recall), while **Cats** and **Birds** proved to be the most difficult classes to distinguish correctly.

| Category | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| ✈️ Airplane | 0.75 | 0.76 | 0.75 | 1000 |
| 🚗 Automobile | 0.76 | 0.96 | 0.85 | 1000 |
| 🐦 Bird | 0.74 | 0.60 | 0.66 | 1000 |
| 🐱 Cat | 0.62 | 0.58 | 0.60 | 1000 |
| 🦌 Deer | 0.71 | 0.78 | 0.74 | 1000 |
| 🐶 Dog | 0.70 | 0.68 | 0.69 | 1000 |
| 🐸 Frog | 0.82 | 0.83 | 0.82 | 1000 |
| 🐎 Horse | 0.84 | 0.75 | 0.79 | 1000 |
| 🚢 Ship | 0.83 | 0.89 | 0.86 | 1000 |
| 🚚 Truck | 0.87 | 0.80 | 0.83 | 1000 |
| | | | | |
| **Accuracy** | | | **0.76** | **10000** |
| **Macro Avg** | 0.76 | 0.76 | 0.76 | 10000 |
| **Weighted Avg** | 0.76 | 0.76 | 0.76 | 10000 |

---

## 🖼 Visualizations

The following analysis plots were generated during the evaluation phase:

### Training History
Visualizes the convergence of loss and accuracy for both training and validation sets.
![Training Curves](https://github.com/ansonnchan/cifar10-image-classification/blob/main/imgs/training_curves.png))

### Confusion Matrix
Provides insight into class-level confusion (e.g., misclassifying Cats as Dogs).
![Confusion Matrix](https://github.com/ansonnchan/cifar10-image-classification/blob/main/imgs/confusion_matrix.png)


### Error Analysis: Misclassified Samples
A qualitative look at test images that the model incorrectly predicted.
![Misclassified Samples](https://github.com/ansonnchan/cifar10-image-classification/blob/main/imgs/misclassified_samples.png)
