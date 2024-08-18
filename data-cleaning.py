import os
import gzip
import shutil

# Example function to decompress a gzip file
def decompress_gzip(input_file, output_file):
    with gzip.open(input_file, 'rb') as f_in:
        with open(output_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

# Example function to clean text
def clean_text(text):
    # Remove unwanted characters, normalize text
    text = text.lower()
    text = ''.join(c for c in text if c.isalnum() or c.isspace())
    return text

# Example usage
input_gzip_file = 'Dataset/Input/robotstxt.paths.gz'
output_text_file = 'Dataset/Output/large_text_file.txt'

# Decompress file
decompress_gzip(input_gzip_file, output_text_file)

# Read and clean text data
with open(output_text_file, 'r') as file:
    text_data = file.read()
    cleaned_text = clean_text(text_data)

# Process cleaned_text with compression algorithms
