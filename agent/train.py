import os
import numpy as np 
import time
import torch
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import DataLoader 
from chess import pgn 
from transform_func import create_input_for_nn, encode_moves
from dataset import ChessDataset
from model import ChessModel
from tqdm import tqdm  
import pickle



def load_pgn(file_path):
    games = []
    with open(file_path, 'r') as pgn_file:
        while True:
            game = pgn.read_game(pgn_file)
            if game is None:
                break
            games.append(game)
    return games

def load_games():
    files = [file for file in os.listdir("../data/Lichess Elite Database") if file.endswith(".pgn")]
    LIMIT_OF_FILES = min(len(files), 28)
    games = []
    i = 1
    for file in tqdm(files):
        games.extend(load_pgn(f"../data/Lichess Elite Database/{file}"))
        if i >= LIMIT_OF_FILES:
            break
        i += 1
    return games
def convert_data_to_tensors():
    games=load_games()
    X, y = create_input_for_nn(games)
    X = X[0:2500000]
    y = y[0:2500000]
    y, move_to_int = encode_moves(y)
    num_classes = len(move_to_int)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    return X,y,num_classes,move_to_int
def train_model():
    X,y,num_classes,move_to_int=convert_data_to_tensors()
    dataset = ChessDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Model Initialization
    model = ChessModel(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    num_epochs = 50
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
            optimizer.zero_grad()

            outputs = model(inputs)  # Raw logits

            # Compute loss
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            running_loss += loss.item()
        end_time = time.time()
        epoch_time = end_time - start_time
        minutes: int = int(epoch_time // 60)
        seconds: int = int(epoch_time) - minutes * 60
        print(f'Epoch {epoch + 1 + 50}/{num_epochs + 1 + 50}, Loss: {running_loss / len(dataloader):.4f}, Time: {minutes}m{seconds}s')
    torch.save(model.state_dict(), "../models/TORCH_100EPOCHS.pth")
    with open("../models/heavy_move_to_int", "wb") as file:
        pickle.dump(move_to_int, file)


if __name__== "__main__":
    train_model()