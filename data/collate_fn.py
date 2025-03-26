from torch.nn.utils.rnn import pad_sequence
import torch

def collate_fn(batch, max_len=256):
    # Filter out empty sequences and truncate long sequences
    batch = [item[:max_len] for item in batch if item.size(0) > 0]  # Truncate to max_len
    
    # If the batch is empty after filtering, return a tensor of zeros (or an empty tensor)
    if len(batch) == 0:
        return torch.zeros(1, 1)
    
    # Otherwise, pad the sequences to the same length
    padded_batch = pad_sequence(batch, batch_first=True, padding_value=0)
    
    # Ensure that all sequences are of the same length
    if padded_batch.size(1) < max_len:
        padding = torch.zeros(padded_batch.size(0), max_len - padded_batch.size(1)).to(padded_batch.device)
        padded_batch = torch.cat((padded_batch, padding), dim=1)
    
    return padded_batch
