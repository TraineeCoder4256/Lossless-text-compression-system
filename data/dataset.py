import torch
from torch.utils.data import Dataset
from datasets import load_dataset  # Import Hugging Face's datasets library
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

class WikiTextByteDataset(Dataset):
    def __init__(self, split='train', tokenizer=None):
        # Load the dataset from Hugging Face
        self.dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Get the text data and tokenize
        text = self.dataset[idx]['text']
        byte_sequence = self.text_to_bytes(text)
        return torch.tensor(byte_sequence)
    
    def text_to_bytes(self, text):
        return list(text.encode('utf-8'))
    
    

# Example of creating a dataset instance
train_dataset = WikiTextByteDataset(split='train')
print(train_dataset[0])  # Print first example (in byte format)
