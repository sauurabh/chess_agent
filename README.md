# Chess AI Agent

## Overview
This project implements a Chess AI model using deep learning techniques. It leverages convolutional neural networks (CNNs) in PyTorch to analyze chess positions and predict the best moves. The model is trained on a dataset of high-level chess games extracted from PGN files.

## Features
- **Data Preprocessing:** Converts chess positions into a 13-channel matrix representation.
- **Model Architecture:** CNN-based neural network for move prediction.
- **Training Pipeline:** Uses PyTorch with CrossEntropy loss and Adam optimizer.
- **GPU Support:** Optimized for training on CUDA-enabled devices.
- **Move Encoding:** Translates chess moves into numerical format for model training.

## Technologies Used
- Python
- PyTorch
- NumPy
- Python-Chess
- tqdm
- Pickle

## Installation
```sh
# Clone the repository
git clone https://github.com/your-username/chess-ai.git
cd chess-ai

# Install dependencies
pip install torch numpy tqdm python-chess
```

## Usage
### Training the Model
To train the model, run the following command:
```sh
python train.py
```
The training script will:
1. Load chess games from PGN files.
2. Convert board states to numerical matrices.
3. Train the CNN model using PyTorch.
4. Save the trained model and move encoder.

### Predicting Moves
To use the trained model for move prediction, implement the following:
```python
import torch
from model import ChessModel

model = ChessModel(num_classes=<num_classes>)
model.load_state_dict(torch.load("models/TORCH_100EPOCHS.pth"))
model.eval()
```

## Project Structure
```
├── data/                      # Directory containing PGN files
├── models/                    # Trained models & encoders
├── dataset.py                  # Custom PyTorch dataset class
├── model.py                    # CNN architecture
├── train.py                    # Training script
├── transform_func.py           # Data transformation utilities
├── README.md                   # Project documentation
```


