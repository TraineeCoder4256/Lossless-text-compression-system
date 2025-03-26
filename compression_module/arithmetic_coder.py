import torch
class ArithmeticCoder:
    def __init__(self, precision=32):
        """
        Initialize the arithmetic coder.
        Args:
            precision (int): The precision for arithmetic coding (default: 32).
        """
        self.precision = precision  # Precision for arithmetic coding

    def encode(self, input_ids, probabilities):
        """
        Encode a sequence of input IDs using arithmetic coding.
        Args:
            input_ids (torch.Tensor): Input sequence of byte values (0–255).
            probabilities (torch.Tensor): Probability distribution for each byte.
        Returns:
            float: Encoded fractional number.
        """
        low, high = 0.0, 1.0  # Initialize the interval
        for i, symbol in enumerate(input_ids):
            # Get the cumulative probabilities for the current symbol
            cum_prob_low = probabilities[i, :symbol].sum().item()
            cum_prob_high = cum_prob_low + probabilities[i, symbol].item()

            # Update the interval
            range_size = high - low
            high = low + range_size * cum_prob_high
            low = low + range_size * cum_prob_low

            # Debug: Print the interval for each symbol
            print(f"Symbol: {symbol}, Interval: [{low}, {high})")

        # Return the midpoint of the final interval
        return (low + high) / 2

    def decode(self, encoded_value, probabilities, sequence_length):
        """
        Decode a sequence from an encoded fractional number.
        Args:
            encoded_value (float): Encoded fractional number.
            probabilities (torch.Tensor): Probability distribution for each byte.
            sequence_length (int): Length of the original sequence.
        Returns:
            torch.Tensor: Decoded sequence of byte values.
        """
        low, high = 0.0, 1.0  # Initialize the interval
        decoded_ids = []
        for i in range(sequence_length):
            # Find the symbol whose probability range contains the encoded value
            for symbol in range(probabilities.size(-1)):
                cum_prob_low = probabilities[i, :symbol].sum().item()
                cum_prob_high = cum_prob_low + probabilities[i, symbol].item()

                # Calculate the symbol's interval
                symbol_low = low + (high - low) * cum_prob_low
                symbol_high = low + (high - low) * cum_prob_high

                if symbol_low <= encoded_value < symbol_high:
                    decoded_ids.append(symbol)
                    # Update the interval
                    low, high = symbol_low, symbol_high
                    # Debug: Print the interval for each symbol
                    print(f"Symbol: {symbol}, Interval: [{low}, {high})")
                    break

        return torch.tensor(decoded_ids, dtype=torch.long)

    def decode_symbol(self, encoded_value, probabilities):
        """
        Decode the next symbol from an encoded fractional number.
        Args:
            encoded_value (float): Encoded fractional number.
            probabilities (torch.Tensor): Probability distribution for the next symbol (shape: [vocab_size]).
        Returns:
            int: Decoded symbol.
        """
        low, high = 0.0, 1.0  # Initialize the interval
        for symbol in range(probabilities.size(0)):  # Iterate over the vocabulary size
            # Get the cumulative probabilities for the current symbol
            cum_prob_low = probabilities[:symbol].sum().item()
            cum_prob_high = cum_prob_low + probabilities[symbol].item()

            # Calculate the symbol's interval
            symbol_low = low + (high - low) * cum_prob_low
            symbol_high = low + (high - low) * cum_prob_high

            if symbol_low <= encoded_value < symbol_high:
                return symbol  # Return the decoded symbol

        return -1  # Should not happen if probabilities are valid