import torch
from model.transformer import CustomTransformerModel
from compression_module.arithmetic_coder import ArithmeticCoder
class DecompressionModel:
    def __init__(self, model, coder):
        self.model = model
        self.coder = coder

    def decompress(self, encoded_value, sequence_length):
        input_ids = torch.tensor([0], dtype=torch.long).unsqueeze(0)  # Start token
        logits = self.model(input_ids)  # Get logits from the model
        probabilities = logits.exp()  # Convert log-probabilities to probabilities
        decoded_sequence = self.coder.decode(encoded_value, probabilities.squeeze(1
                                                                                ), sequence_length)
        return decoded_sequence
