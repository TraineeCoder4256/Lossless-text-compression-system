import torch

def save_model(model, filepath):
    """
    Save the model's state dictionary to a file.
    """
    torch.save(model.state_dict(), filepath)

def load_model(model, filepath):
    """
    Load the model's state dictionary from a file.
    """
    model.load_state_dict(torch.load(filepath))
    model.eval()
