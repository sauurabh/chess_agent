from chess import Board, pgn
from agent.transform_func import board_to_matrix
import torch
from agent.model import ChessModel
import pickle
import numpy as np


def prepare_input(board):
    matrix = board_to_matrix(board)
    X_tensor = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)
    return X_tensor

def main(board):
    move_to_int=None
    with open("models/move_to_int", "rb") as file:
        move_to_int = pickle.load(file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the model
    model = ChessModel(num_classes=len(move_to_int))
    model.load_state_dict(torch.load("models/TORCH_100EPOCHS.pth"))
    model.to(device)
    model.eval()
    int_to_move = {v: k for k, v in move_to_int.items()}

    X_tensor = prepare_input(board).to(device)
    
    with torch.no_grad():
        logits = model(X_tensor)
    
    logits = logits.squeeze(0)  # Remove batch dimension
    
    probabilities = torch.softmax(logits, dim=0).cpu().numpy()  # Convert to probabilities
    legal_moves = list(board.legal_moves)
    legal_moves_uci = [move.uci() for move in legal_moves]
    sorted_indices = np.argsort(probabilities)[::-1]
    for move_index in sorted_indices:
        move = int_to_move[move_index]
        if move in legal_moves_uci:
            return move
    
    return None

if __name__=="__main__":
    main()