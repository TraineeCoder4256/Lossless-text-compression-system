import torch
from torch.utils.data import DataLoader
from model.transformer import CustomTransformerModel
from data.dataset import WikiTextByteDataset
import torch.nn.functional as F

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    batch_idx=0
    with torch.no_grad():
        for batch in test_loader:
            # Ensure the batch is of type LongTensor
            print(f"Evaluating batch {batch_idx}/{len(test_loader)}")
            batch = batch.long().to(next(model.parameters()).device)

            # Forward pass
            output = model(batch)

            # Get predicted tokens
            _, predicted = torch.max(output, dim=-1)

            # Flatten the batch and predicted tensors for comparison
            total += batch.numel()
            correct += (predicted == batch).sum().item()
            batch_idx+=1

    accuracy = correct / total
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
