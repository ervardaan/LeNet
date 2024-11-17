# LeNet

![LeNet-5 Architecture](https://upload.wikimedia.org/wikipedia/commons/3/35/LeNet-5_architecture.svg "LeNet-5 Architecture")

# LeNet Implementation

This repository contains an implementation of the LeNet convolutional neural network (CNN) architecture, primarily designed for digit recognition tasks such as the MNIST dataset. It is a classic architecture that paved the way for modern deep learning methods.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Experiments](#experiments)
- [Results](#results)
- [Project Structure](#project-structure)
- [License](#license)

## Overview

LeNet, developed by Yann LeCun in 1998, is one of the earliest convolutional neural networks. It consists of convolutional layers, subsampling (pooling) layers, and fully connected layers, making it a fundamental architecture in the field of deep learning.

This project explores:
1. Training and evaluating LeNet on the MNIST dataset.
2. Understanding the effects of various hyperparameters on performance.
3. Implementing the architecture in Python using deep learning frameworks.

## Features

- **CNN Architecture**: Includes convolutional, pooling, and fully connected layers.
- **MiniPlaces Dataset**: Scene classification dataset with diverse categories.
- **Customization**: Easily modify parameters like learning rate, batch size, and epochs.
- **Documentation**: Detailed write-up in [hw7.pdf](https://github.com/ervardaan/LeNet/blob/main/docs/hw7.pdf).

## Installation

### Prerequisites
- Python 3.x
- Deep learning library (e.g., TensorFlow or PyTorch)
- NumPy
- Matplotlib (for visualizations)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/ervardaan/LeNet.git
   cd LeNet

2. Install dependencies:
bash
pip install -r requirements.txt
Usage
### Training
To train the model on the MNIST dataset, run:

bash
python train.py --epochs 10 --batch_size 64 --learning_rate 0.01
### Evaluation
To evaluate the trained model, run:

bash
python evaluate.py --model_path saved_models/lenet.pth
### Visualization
To visualize training metrics:

bash
python visualize.py --log_dir logs/

### Customization
You can adjust the following hyperparameters via command-line arguments:

--epochs: Number of training epochs.
--batch_size: Size of each training batch.
--learning_rate: Learning rate for optimization.

### Experiments
This project explores:

Training on MNIST: Evaluating the model's ability to classify handwritten digits.
Hyperparameter Tuning: Assessing the effects of learning rate, batch size, and other parameters.
Performance Analysis: Comparing training and validation accuracies, and visualizing losses.
For detailed experiment results and observations, see the project write-up.

Results
The model achieves:

Training Accuracy: ~99%
Validation Accuracy: ~98%
Detailed results, including loss curves and confusion matrices, are available in the project write-up.

Project Structure
bash
Copy code
LeNet/
├── docs/
│   └── hw7.pdf             # Project write-up
├── lenet/
│   ├── __init__.py         # Package initialization
│   ├── model.py            # LeNet architecture
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation script
│   ├── visualize.py        # Visualization script
│   └── utils.py            # Helper functions
├── data/                   # MNIST dataset (or link to download)
├── saved_models/           # Trained models
├── logs/                   # Training logs
├── requirements.txt        # Python dependencies
└── README.md               # Project description
License
This project is licensed under the MIT License. See the LICENSE file for details.

vbnet
Copy code

This `README.md` provides a comprehensive overview of the project, making it easy for others to understand and use your repository. Let me know if there are specific additions or changes you'd like!






