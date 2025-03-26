import torch
from model.transformer import CustomTransformerModel
from compression_module.arithmetic_coder import ArithmeticCoder
from compression_module.compression_engine import CompressionModel
from compression_module.decompression_engine import DecompressionModel

# Load your trained transformer model (make sure it's loaded correctly)
model = CustomTransformerModel()
model.load_state_dict(torch.load('saved_model.pth'))  # Load your trained model
model.eval()  # Don't forget to set the model to evaluation mode

# Initialize the arithmetic coder and compression model
coder = ArithmeticCoder(precision=32)
compression_model = CompressionModel(model=model, coder=coder)
decompression_model = DecompressionModel(model=model, coder=coder)

# Example test sequence (as a list of byte values, e.g., ASCII values)
input_sequence = [104, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100]  # "hello world"

# Convert the input sequence to a tensor
input_tensor = torch.tensor(input_sequence, dtype=torch.long)  # Add batch dimension

# Compress the input sequence
encoded_value = compression_model.compress(input_tensor)
print(f"Encoded (Compressed) Value: {encoded_value}")

# Decompress the encoded value back to the original sequence
'''sequence_length = len(input_sequence)
decoded_sequence = decompression_model.decompress(encoded_value, sequence_length)
print(f"Decoded (Decompressed) Sequence: {decoded_sequence}")'''
