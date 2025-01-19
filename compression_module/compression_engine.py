import torch
from model.transformer import CustomTransformerModel
from compression_module.arithmetic_coder import ArithmeticCoder

class CompressionModel:
    def __init__(self, model, coder):
        self.model = model  # Trained transformer model
        self.coder = coder  # Arithmetic coder

    def compress(self, input_sequence):
        """
        Compress the input sequence using the transformer model and arithmetic coding.
        Args:
            input_sequence (list or torch.Tensor): Sequence of byte values (e.g., text or bytes).
        Returns:
            float: Encoded value (compressed representation).
        """
        input_tensor = torch.tensor(input_sequence).unsqueeze(0)  # Add batch dimension

        # Get model output (log probabilities)
        model_output = self.model(input_tensor)
        probabilities = model_output.exp()  # Convert log-probabilities to probabilities

        # Perform arithmetic encoding
        encoded_value = self.coder.encode(input_tensor.squeeze(0), probabilities.squeeze(0))
        return encoded_value

