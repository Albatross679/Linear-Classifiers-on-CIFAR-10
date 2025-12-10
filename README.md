# ECE 5460 Assignment 4: Linear Classifiers on CIFAR-10

This project implements and trains a linear classifier for image classification on the CIFAR-10 dataset. The project explores the fundamentals of linear classification in computer vision using PyTorch.

## Project Overview

This assignment focuses on:
- Loading and preprocessing the CIFAR-10 dataset
- Implementing a linear classifier using PyTorch
- Training the classifier on CIFAR-10 images
- Evaluating model performance with per-class accuracy analysis
- Visualizing training curves and results

## Dataset

The project uses the **CIFAR-10 dataset**, which contains:
- 50,000 training images
- 10,000 test images
- 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- 32x32 color images

## Project Structure

```
Project 4/
├── ece5460_assignment4.ipynb    # Main Jupyter notebook with complete implementation
├── train_classifier.py           # Standalone training script
├── requirements.txt              # Python dependencies
├── linear_classifier.pth         # Trained model weights
├── classifier_results.png        # Visualization of results
└── README.md                     # This file
```

## Requirements

- Python 3.x
- PyTorch 2.9.1
- torchvision 0.24.1
- NumPy
- Matplotlib
- Jupyter Notebook

See `requirements.txt` for the complete list of dependencies.

## Installation

1. Clone or download this repository

2. Create a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Jupyter Notebook

1. Open the notebook:
```bash
jupyter notebook ece5460_assignment4.ipynb
```

2. Run all cells to:
   - Load and visualize CIFAR-10 data
   - Train the linear classifier
   - Evaluate on test set
   - Generate visualizations

### Option 2: Standalone Script

Run the training script directly:
```bash
python train_classifier.py
```

This will:
- Download CIFAR-10 dataset (if not already present)
- Train the linear classifier for 25 epochs
- Evaluate on the test set
- Print per-class and overall accuracy
- Save the trained model to `linear_classifier.pth`

## Model Architecture

The linear classifier consists of:
- **Input**: Flattened 32×32×3 = 3,072 pixel values
- **Output**: 10 class logits (one for each CIFAR-10 class)
- **Architecture**: Single fully connected linear layer

```python
class LinearClassifier(torch.nn.Module):
    def __init__(self, input_dim=32*32*3, output_dim=10):
        super(LinearClassifier, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
```

## Training Details

- **Optimizer**: SGD with learning rate 0.001 and momentum 0.9
- **Loss Function**: Cross-Entropy Loss
- **Epochs**: 25
- **Batch Size**: 10
- **Device**: Automatically uses CUDA GPU, Apple MPS, or CPU based on availability

## Results

The model achieves:
- Overall test accuracy (varies by run, typically ~30-40%)
- Per-class accuracy breakdown
- Comparison with random guessing baseline (10%)

Results are visualized in `classifier_results.png`, showing:
- Training and validation loss curves
- Training and validation accuracy curves
- Comparison with baseline performance

## Features

- **Automatic device detection**: Uses GPU (CUDA/MPS) if available, falls back to CPU
- **Comprehensive evaluation**: Per-class accuracy analysis
- **Visualization**: Training curves and performance comparisons
- **Model persistence**: Trained models are saved for later use

## Analysis

The project includes analysis of:
1. **Baseline comparison**: Performance vs. random guessing (10% for 10 classes)
2. **Class performance**: Best and worst performing classes
3. **Performance gap**: Analysis of class-wise accuracy differences

## Notes

- The linear classifier provides a baseline for image classification
- Better performance can be achieved with convolutional neural networks (CNNs)
- The model demonstrates the importance of feature extraction in computer vision

## Course Information

- **Course**: ECE 5460 - Image Processing
- **Institution**: OSU (The Ohio State University)
- **Semester**: AU25 (Autumn 2025)

## License

This project is part of an academic assignment. Please use responsibly and in accordance with your institution's academic integrity policies.
