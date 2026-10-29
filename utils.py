import torch
import os
import time


def save_checkpoint(state, filename):
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(filename, model, optimizer=None):
    checkpoint = torch.load(filename, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Checkpoint loaded from {filename}")
    return checkpoint


class Logger:
    def __init__(self, log_file, append=False):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        mode = 'a' if append else 'w'
        with open(log_file, mode) as f:
            f.write(f"Log {'resumed' if append else 'started'} at {time.ctime()}\n")

    def log(self, message):
        with open(self.log_file, 'a') as f:
            f.write(f"{time.ctime()}: {message}\n")
        print(message)