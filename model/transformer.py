import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class CustomTransformerModel(nn.Module):
    def __init__(self, vocab_size=256, embedding_dim=512, num_layers=6, num_heads=8, hidden_dim=2048, max_seq_len=256, dropout=0.1):
        super(CustomTransformerModel, self).__init__()

        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(embedding_dim, max_seq_len)

        # Transformer Encoder Layers
        self.encoder_layers = nn.ModuleList(
            [nn.TransformerEncoderLayer(
                d_model=embedding_dim, 
                nhead=num_heads, 
                dim_feedforward=hidden_dim, 
                dropout=dropout
            ) for _ in range(num_layers)]
        )
        
        # Transformer Encoder
        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layers[0],  # Corrected from 'layer' to 'encoder_layer'
            num_layers=num_layers
        )

        # Output Layer (to predict next byte)
        self.output_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        # Add positional encoding to input
        x = self.embedding(x) + self.positional_encoding(x.size(1)).to(x.device)

        # Pass through encoder layers
        x = self.encoder(x)

        # Predict next token probabilities
        x = self.output_layer(x)
        return F.log_softmax(x, dim=-1)

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_seq_len=256):
        super(PositionalEncoding, self).__init__()

        # Create the positional encodings
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * -(math.log(10000.0) / embedding_dim))
        pe = torch.zeros(max_seq_len, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, seq_len):
        return self.pe[:, :seq_len]
