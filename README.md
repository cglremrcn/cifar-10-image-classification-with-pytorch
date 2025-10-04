# CIFAR-10 Image Classification with PyTorch

This repository contains a PyTorch implementation for image classification on the CIFAR-10 dataset using a ResNet-18 model. It is structured for easy training, evaluation, and predictions.

## Project Structure

```
cifar-10-image-classification-with-pytorch/
│
├── data/                 # Dataset storage (not included in repo)
├── saved_models/         # Saved model weights (not included in repo)
├── src/                  # Source code
│   ├── data.py           # Functions to load and preprocess CIFAR-10 dataset
│   ├── model.py          # Model definition (ResNet-18)
│   ├── utils.py          # Utility functions (train, test, accuracy, predictions)
│   └── config.py         # Configuration (batch size, learning rate, epochs, etc.)
├── train.py              # Script to train the model
├── plot.py               # Script for plotting results (confusion matrix, metrics)
├── predict.py            # Script to make predictions on new images
├── README.md
└── .gitignore
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/cifar-10-image-classification-with-pytorch.git
cd cifar-10-image-classification-with-pytorch
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv venv
# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux / Mac:
source venv/bin/activate

pip install -r requirements.txt
```

> Recommended dependencies: `torch`, `torchvision`, `torchmetrics`, `mlxtend`, `matplotlib`, `tqdm`

## Usage

### Train the model

```bash
python train.py
```

This will train the ResNet-18 model on CIFAR-10 and save the weights to `saved_models/resnet18_cifar10.pth`.

### Evaluate and plot results

```bash
python plot.py
```

Generates confusion matrix and metrics for the trained model.

### Make predictions

```bash
python predict.py --image path_to_image
```

Predicts the class of a given image using the trained model.

## Notes

* The `data/` and `saved_models/` folders are **not included** in the repository. You need to download CIFAR-10 and create `saved_models/` manually.
* Make sure your device supports CUDA if you want GPU acceleration.

## License

This project is open-source and available under the MIT License.
