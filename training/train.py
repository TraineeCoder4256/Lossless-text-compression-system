import torch
from torch.utils.data import DataLoader
from model.transformer import CustomTransformerModel  # Your custom transformer model
from data.dataset import WikiTextByteDataset
from model.utils import save_model, load_model
import os,time

# Hyperparameters
BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
SAVE_PATH = 'saved_model.pth'  # Path to save the model

# Set up DataLoader for training
def get_data_loaders():
    train_dataset = WikiTextByteDataset(split='train')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    return train_loader

# Initialize the model
def initialize_model():
    model = CustomTransformerModel()
    return model

# Training loop
def start_training(model, train_loader):
    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        print(f"Starting Epoch {epoch+1}/{NUM_EPOCHS}")  # Debug: Epoch Start
        #epoch_start_time = time.time()  # Track time at the start of the epoch
        model.train()
        total_loss = 0
        batch_idx=0
        for batch in train_loader:
            print(f"-Processing Batch {batch_idx}/{len(train_loader)}")  # Debug: Batch Start
            optimizer.zero_grad()
            
            # Forward pass
            output = model(batch)
            
            # Calculate loss
            loss = criterion(output.view(-1, output.size(-1)), batch.view(-1))
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            optimizer.step()
            batch_idx+=1

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {total_loss/len(train_loader)}")

    # After training, save the model
    save_model(model, SAVE_PATH)

    print(f"Training completed and model saved to {SAVE_PATH}")

# # Example to load a pre-trained model and run inference
def load_trained_model():
    model = CustomTransformerModel()
    model = load_model(model, SAVE_PATH)
    return model

# Training loop with checkpoint saving
# def start_training(model, train_loader):
#     # Define loss function and optimizer
#     criterion = torch.nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

#     # Training loop
#     for epoch in range(NUM_EPOCHS):
#         print(f"Starting Epoch {epoch+1}/{NUM_EPOCHS}")  # Debug: Epoch Start
#         model.train()
#         total_loss = 0
#         batch_idx = 0
#         for batch in train_loader:
#             print(f"-Processing batch {batch_idx}/{len(train_loader)}")  # Debug: Batch Start
#             optimizer.zero_grad()

#             # Forward pass
#             output = model(batch)

#             # Calculate loss
#             loss = criterion(output.view(-1, output.size(-1)), batch.view(-1))
#             total_loss += loss.item()

#             # Backward pass
#             loss.backward()
#             optimizer.step()
#             batch_idx += 1

#         print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {total_loss / len(train_loader)}")

#         # Save checkpoint after each epoch
#         checkpoint_path = f"checkpoint_epoch_{epoch+1}.pth"
#         torch.save({
#             'epoch': epoch + 1,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'loss': total_loss / len(train_loader)
#         }, checkpoint_path)
#         print(f"Checkpoint saved: {checkpoint_path}")

#     # Save the final model
#     torch.save(model.state_dict(), SAVE_PATH)
#     print(f"Training completed and final model saved to {SAVE_PATH}")


# def load_trained_model():
#     model = CustomTransformerModel()
#     model = load_model(model, SAVE_PATH)
#     return model