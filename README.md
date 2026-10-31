# Handwritten Digit Recognition using PyTorch CNN

A deep learning project that classifies handwritten digits (0–9) using a Convolutional Neural Network (CNN) built with **PyTorch**. Developed for the [Kaggle Digit Recognizer competition](https://www.kaggle.com/competitions/digit-recognizer).

 [![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/) ![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?logo=pytorch) ![Kaggle](https://img.shields.io/badge/Kaggle-Digit%20Recognizer-20BEFF?logo=kaggle) 

## Project Overview

The objective is to recognize handwritten digits from 28×28 grayscale images. The model takes an image as input and predicts one of ten possible classes (0–9).

Instead of using traditional machine learning algorithms, this project implements a CNN from scratch using PyTorch, exploring convolutional layers, batch normalization, pooling, dropout, and data augmentation.

## Results

| Metric | Result |
|---|---:|
| Best Kaggle public leaderboard accuracy | **99.392%** |
| Number of training images | 42,000 |
| Number of test images | 28,000 |
| Image dimensions | 28 × 28 |
| Number of classes | 10 |

## Dataset

The project uses the [Kaggle Digit Recognizer dataset](https://www.kaggle.com/competitions/digit-recognizer/data), based on the MNIST handwritten digit dataset.

- **Training set:** 42,000 labeled images.
- **Test set:** 28,000 unlabeled images.
- **Image dimensions:** 28 × 28 pixels.
- **Pixel values:** 0–255, normalized to the range [0, 1].
- **Target:** Digit labels from 0 to 9.

Each image is represented by 784 pixel values in the original CSV files.

## Model Architecture

A custom CNN was implemented using `torch.nn.Module`.

The network consists of two main components:

### 1. Feature Extraction

| Layer | Configuration |
|---|---|
| Convolution 1 | 1 → 32 channels, 3×3 kernel |
| Batch Normalization | 32 channels |
| Activation | ReLU |
| Max Pooling | 2×2 |
| Convolution 2 | 32 → 64 channels, 3×3 kernel |
| Batch Normalization | 64 channels |
| Activation | ReLU |
| Max Pooling | 2×2 |

### 2. Classification

| Layer | Configuration |
|---|---|
| Flatten | 64 × 7 × 7 → 3136 |
| Fully Connected 1 | 3136 → 128 |
| Activation | ReLU |
| Dropout | 0.4 |
| Fully Connected 2 | 128 → 64 |
| Activation | ReLU |
| Dropout | 0.4 |
| Output Layer | 64 → 10 |

The final layer produces 10 logits, one for each digit class.

## Data Preprocessing and Augmentation

The following preprocessing steps were applied:

- Converted CSV pixel data into PyTorch tensors.
- Reshaped images into `[1, 28, 28]` format.
- Normalized pixel intensities by dividing by 255.
- Split the training data into training and validation sets (80:20).

To improve generalization, random affine transformations were also explored:

- Random rotation: up to ±7°.
- Random translation: up to 8% horizontally and vertically.
- Zero padding/fill for newly exposed pixels.

These transformations help the model recognize digits despite small variations in their orientation and position.

## Training

The model was trained using:

- **Framework:** PyTorch
- **Loss Function:** CrossEntropyLoss
- **Optimizer:** SGD and AdamW (experimented with both)
- **Regularization:** Dropout and weight decay
- **Validation:** Held-out validation dataset

Multiple experiments were conducted to understand the effects of optimizer selection, dropout, and data augmentation on model performance.

## Key Learnings

Through this project, I explored and practiced:

- Building custom CNN architectures using PyTorch.
- Understanding convolution, pooling, and feature extraction.
- Implementing custom PyTorch `Dataset` and `DataLoader` classes.
- Training and validating neural networks.
- Working with CrossEntropyLoss and classification logits.
- Understanding overfitting and generalization.
- Experimenting with optimizers and regularization techniques.
- Applying data augmentation to improve model robustness.
- Generating Kaggle-compatible prediction submissions.

## Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Kaggle

## Repository Structure

```text
Digit-Recognizer/
│
├── notebook.ipynb       # Data preprocessing, model development and training
├── submission.csv       # Kaggle prediction submission
└── README.md
```

*Adjust the filenames above to match the actual files in the repository.*

## How to Run

1. Clone the repository:

```bash
git clone <your-repository-url>
cd Digit-Recognizer
```

2. Install the dependencies:

```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib
```

3. Download the dataset from the [Kaggle competition page](https://www.kaggle.com/competitions/digit-recognizer/data).

4. Place the dataset CSV files in the appropriate directory.

5. Open and run the notebook to reproduce the preprocessing, training, evaluation, and prediction workflow.

## Competition

[Kaggle – Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer)

---

**Author:** [Your GitHub Username](https://github.com/your-username)
