import torch
from torch.utils.data import DataLoader
from training.train import start_training
from data.dataset import WikiTextByteDataset
from model.transformer import CustomTransformerModel
from configs.config import config
from training.evaluate import evaluate_model
from data.collate_fn import collate_fn
from model.utils import save_model, load_model

# Load data using the WikiTextByteDataset class
train_dataset = WikiTextByteDataset(split='train')
test_dataset = WikiTextByteDataset(split='test')

train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False,collate_fn=collate_fn)

# Initialize model
model = CustomTransformerModel(
    vocab_size=config['vocab_size'],
    embedding_dim=config['embedding_dim'],
    num_layers=config['num_layers'],
    num_heads=config['num_heads'],
    hidden_dim=config['hidden_dim'],
    max_seq_len=config['seq_length']
)

# Start training
start_training(model, train_loader)
model = load_model(model, 'saved_model.pth')
evaluate_model(model, test_loader)
