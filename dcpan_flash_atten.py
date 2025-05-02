import torch

# path of train and test .tsv
path = ""
import os
import pandas as pd
import numpy as np

import sys
from estimator import FeatureExtractor
import torch.nn as nn
import torch.optim as optim
import math
from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F

import builtins
# Override the built-in print function
def print(*args, **kwargs):
    kwargs.setdefault('flush', True)  # Set flush=True by default
    builtins.print(*args, **kwargs)


torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("CUDA available:", torch.cuda.is_available())
print("PyTorch version:", torch.__version__)

# CHANGE OUTPUT CHANNELS
batch_size = 2
dcpan_channels = 8
attention_heads = 8
lr = 0.001
num_epochs = 5
accumulation_steps = 4

# Choose evaluation mode
compare_features = True  # Set to True to compare feature similarities, False to compare age predictions

print(f"Batch Size: {batch_size}")
print(f"DCPAN Channels: {dcpan_channels}")
print(f"Attention Heads: {attention_heads}")
print(f"LR: {lr}")
print(f"Epochs: {num_epochs}")
print(f"Accumulation steps: {accumulation_steps}")

if torch.cuda.is_available():
    device = torch.device("cuda")  # Use the first available GPU
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")  # Fallback to CPU
    print("CUDA is not available. Using CPU.")

# print("Device name:", torch.cuda.get_device_name(0))

if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f"Number of available CUDA devices: {device_count}")
    for i in range(device_count):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available. Using CPU.")





def save_filtered_data(path, dataset):
    df = pd.read_csv(os.path.join(path, "data", dataset + ".tsv"), sep="\t")
    x_arr = np.load(os.path.join(path, "data", dataset + ".npy"), mmap_mode="r")
    filtered_df = df[df["site"] == 1]
    y_filtered = filtered_df[["age", "site"]].values
    x_filtered = x_arr[filtered_df.index] 
    filtered_tsv_path = os.path.join(path, "data", f"{dataset}_site_1.tsv")
    filtered_npy_path = os.path.join(path, "data", f"{dataset}_site_1.npy")
    filtered_df.to_csv(filtered_tsv_path, sep="\t", index=False)
    np.save(filtered_npy_path, x_filtered)
    
    print(f"Filtered data saved to: {filtered_tsv_path}")
    print(f"Filtered x_arr saved to: {filtered_npy_path}")

save_filtered_data(path, "train")
save_filtered_data(path, "test")


# Need to specify where to find the brain and GM masks using env variables.
os.environ["VBM_MASK"] = "./cat12vbm_space-MNI152_desc-gm_TPM.nii.gz"
os.environ["QUASIRAW_MASK"] = "./quasiraw_space-MNI152_desc-brain_T1w.nii.gz"
# %matplotlib inline
from problem import get_train_data, get_test_data, DatasetHelper

train_dataset = DatasetHelper(data_loader=get_train_data)
X_train, y_train = train_dataset.get_data()
test_dataset = DatasetHelper(data_loader=get_test_data)
X_test, y_test = test_dataset.get_data()

class BrainAgeDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        sample = np.expand_dims(self.X_data[idx], axis=0)
        label = self.y_data[idx,0]
        sample = FeatureExtractor(dtype="vbm").transform(sample)
        sample = torch.tensor(sample, dtype=torch.float16)
        sample = sample.squeeze(0)
        sample = sample.squeeze(0)
        label = torch.tensor(label, dtype=torch.float16)
        return sample, label
dataset = BrainAgeDataset(X_train, y_train)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
for inputs, labels in train_loader:
    print(inputs.shape, labels.shape)
    break


class MultiHeadNeighborhoodAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=8, neighborhood_size=3):
        super(MultiHeadNeighborhoodAttentionBlock, self).__init__()
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads
        assert self.head_dim * num_heads == out_channels, "out_channels must be divisible by num_heads"

        # Projections for Q, K, V
        self.query_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        
        # Output projection
        self.out_conv = nn.Conv3d(out_channels, out_channels, kernel_size=1)
        
        # # Relative positional bias
        # self.relative_bias = nn.Parameter(torch.zeros(neighborhood_size**3))

        self.neighborhood_size = neighborhood_size

    def forward(self, x):
        """
        x: Input feature map, shape [B, C, D, H, W]
        """
        B, C, D, H, W = x.shape
        
        # Step 1: Generate Q, K, V projections
        query = self.query_conv(x)  # Shape: [B, C_out, D, H, W]
        key = self.key_conv(x)      # Shape: [B, C_out, D, H, W]
        value = self.value_conv(x)  # Shape: [B, C_out, D, H, W]
        
        # Reshape Q, K, V for multi-head attention
        query = query.view(B, self.num_heads, self.head_dim, D, H, W)
        key = key.view(B, self.num_heads, self.head_dim, D, H, W)
        value = value.view(B, self.num_heads, self.head_dim, D, H, W)
        
        # Step 2: Pad for neighborhoods
        padding = self.neighborhood_size // 2
        key_padded = F.pad(key, (padding, padding, padding, padding, padding, padding), mode='constant', value=0)
        value_padded = F.pad(value, (padding, padding, padding, padding, padding, padding), mode='constant', value=0)

        # Step 3: Extract neighborhoods for K and V
        key_unfold = key_padded.unfold(3, self.neighborhood_size, 1).unfold(4, self.neighborhood_size, 1).unfold(5, self.neighborhood_size, 1)
        value_unfold = value_padded.unfold(3, self.neighborhood_size, 1).unfold(4, self.neighborhood_size, 1).unfold(5, self.neighborhood_size, 1)

        # Reshape to match Q
        key_unfold = key_unfold.contiguous().view(B, self.num_heads, self.head_dim, self.neighborhood_size**3, -1)  # [B, num_heads, head_dim, K*K*K, N]
        value_unfold = value_unfold.contiguous().view(B, self.num_heads, self.head_dim, self.neighborhood_size**3, -1)  # [B, num_heads, head_dim, K*K*K, N]
        query_flat = query.view(B, self.num_heads, self.head_dim, -1)  # [B, num_heads, head_dim, N]

        # print(f"query_flat shape: {query_flat.shape}")
        # print(f"key_unfold shape: {key_unfold.shape}")
        # print(f"value_unfold shape: {value_unfold.shape}")
        # Step 4: Compute attention scores
        # attention_scores = torch.einsum('bhcn,bhckn->bhkn', query_flat, key_unfold)  # [B, num_heads, K*K*K, N]
        # attention_scores = attention_scores + self.relative_bias.view(1, 1, -1, 1)
        # attention_scores = F.softmax(attention_scores, dim=2)  # Normalize along the neighborhood axis

        # # Step 5: Apply attention to values
        # attended_values = torch.einsum('bhkn,bhckn->bhcn', attention_scores, value_unfold)  # [B, num_heads, head_dim, N]
        # attended_values = attended_values.view(B, self.num_heads * self.head_dim, D, H, W)  # Reshape back to original dimensions
        

        # Reshape query_flat to match key_unfold and value_unfold
        query_flat = query_flat.unsqueeze(3)  # Add singleton dimension for neighborhood size
        # Now query_flat shape: [2, 8, 1, 1, 2122945]

        # Step 4: Compute attention scores
        attn_output = F.scaled_dot_product_attention(
            query_flat,  # Shape: [B, num_heads, head_dim, 1, N]
            key_unfold,  # Shape: [B, num_heads, head_dim, K*K*K, N]
            value_unfold,  # Shape: [B, num_heads, head_dim, K*K*K, N]
            is_causal=False
        )
        
        # Remove the singleton dimension
        attn_output = attn_output.squeeze(3)  # Shape: [B, num_heads, head_dim, N]
        
        # Reshape back to original dimensions
        attended_values = attn_output.view(B, self.num_heads * self.head_dim, D, H, W)
        
        # Step 5: Final projection
        output = self.out_conv(attended_values)  # Shape: [B, C_out, D, H, W]
        
        return output
        
        
        
        # attn_output = F.scaled_dot_product_attention(query_flat, key_unfold, value_unfold, is_causal=False) 
        # out = attn_output.transpose(1, 2).contiguous().view(B, -1, self.num_heads * self.head_dim)
        # attended_values = out.view(B, self.num_heads * self.head_dim, D, H, W)
        # # # Step 6: Final projection
        # output = self.out_conv(attended_values)  # Shape: [B, C_out, D, H, W]
        
        # return output


class DCPAN3D(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=4):
        super(DCPAN3D, self).__init__()
        # Dimension-wise 3D Convolutions
        self.depth_conv = nn.Conv3d(out_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.height_conv = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 1, 3), padding=(1, 0, 1))
        self.width_conv = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 1), padding=(1, 1, 0))

        # Multi-Head Neighborhood Attention Blocks
        self.depth_attention = MultiHeadNeighborhoodAttentionBlock(out_channels, out_channels, num_heads)
        self.height_attention = MultiHeadNeighborhoodAttentionBlock(out_channels, out_channels, num_heads)
        self.width_attention = MultiHeadNeighborhoodAttentionBlock(out_channels, out_channels, num_heads)
        
        # Tri-Branch Aggregation
        self.fusion_conv = nn.Conv3d(out_channels * 3, out_channels, kernel_size=1)

        # Position Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, out_channels, 121, 145, 121))  # Adjust size as needed
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # Add position embedding
        x = x + self.pos_embed
        
        # Dimension-wise feature extraction
        depth_features = self.depth_conv(x)
        height_features = self.height_conv(x)
        width_features = self.width_conv(x)
        
        # Apply Multi-Head Neighborhood Attention
        depth_attention = self.depth_attention(depth_features)
        height_attention = self.height_attention(height_features)
        width_attention = self.width_attention(width_features)
        
        # Fuse features
        combined_features = torch.cat([depth_attention, height_attention, width_attention], dim=1)
        fused_features = self.fusion_conv(combined_features)
        return fused_features


# Test predicting with FC layer for DCPAN
class AgePredictor(nn.Module):
    def __init__(self, dcpan3d, out_channels, num_classes=1):
        super(AgePredictor, self).__init__()
        self.dcpan3d = dcpan3d
        
        # Spatial reduction and feature extraction
        self.conv1 = nn.Conv3d(out_channels, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv3d(64, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv3d(32, 16, kernel_size=3, stride=2, padding=1)
        
        # Calculate the size after spatial reduction
        # Original size: 121x145x121
        # After 3 stride-2 convolutions: 16x19x16
        self.reduced_size = 16 * 16 * 19 * 16  # 16 channels * 16x19x16 spatial dimensions
        
        # FC layers for final prediction
        self.fc = nn.Sequential(
            nn.Linear(self.reduced_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # Pass through DCPAN3D
        features = self.dcpan3d(x)
        
        # Spatial reduction and feature extraction
        x = F.relu(self.conv1(features))
        # x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        
        # print(f"Input shape: {features.shape}")
        # print(f"After conv1: {x.shape}")
        x = F.relu(self.conv2(x))
        # print(f"After conv2: {x.shape}")
        x = F.relu(self.conv3(x))
        # print(f"After conv3: {x.shape}")
        
        # Flatten the reduced features
        x = x.view(x.size(0), -1)
        # print(f"Flattened shape: {x.shape}")
        
        # Final prediction
        age_prediction = self.fc(x)
        return age_prediction


print("ONE")
# if torch.isnan(torch.tensor(X_train)).any() or torch.isinf(torch.tensor(X_train)).any():
#     raise ValueError("Input data contains NaN or inf values.")

print("TWO")
# dcpan_3d = DCPAN3D(in_channels=1, out_channels=dcpan_channels).to(device)
dcpan_3d = DCPAN3D(in_channels=1, out_channels=dcpan_channels, num_heads=attention_heads).to(device)
print("TWO Half")
age_predictor = AgePredictor(dcpan_3d, out_channels=dcpan_channels).to(device)

print("THREE")
total_params = sum(p.numel() for p in age_predictor.parameters())
print(f"Total number of parameters in the model: {total_params}")

print("FOUR")
criterion = nn.MSELoss()
optimizer = optim.Adam(age_predictor.parameters(), lr=lr)  # Lower learning rate

# Grad clipping
max_grad_norm = 1.0


# FIX FP16 ERROR Training loop with AMP and gradient clipping
scaler = torch.amp.GradScaler()  # For mixed precision training

print("FIVE")
for epoch in range(num_epochs):
    age_predictor.train()
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device).unsqueeze(1)
        labels = labels.to(device).float()
        
        with torch.amp.autocast(device_type='cuda'):
            outputs = age_predictor(inputs)
            outputs = outputs.view(-1)
            loss = criterion(outputs.squeeze(), labels)
            loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(age_predictor.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# # Create test dataset and loader
# test_dataset = BrainAgeDataset(X_test, y_test)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # No need to shuffle for evaluation

# # Eval
# age_predictor.eval()
# with torch.no_grad():
#     # Compare age predictions
#     total_error = 0
#     total_squared_error = 0
#     num_samples = 0
#     all_outputs = []
#     all_labels = []
    
#     # First collect all predictions from test set
#     for inputs, labels in test_loader:
#         inputs = inputs.to(device).unsqueeze(1)
#         labels = labels.to(device).float()
        
#         with torch.amp.autocast(device_type='cuda'):
#             outputs = age_predictor(inputs)
#             outputs = outputs.view(-1)
#             all_outputs.append(outputs)
#             all_labels.append(labels)
    
#     # Stack all predictions and labels
#     all_outputs = torch.cat(all_outputs, dim=0)
#     all_labels = torch.cat(all_labels, dim=0)
    
#     # Calculate metrics on entire test set
#     errors = torch.abs(all_outputs - all_labels)
#     squared_errors = errors ** 2
#     mae = errors.mean().item()
#     mse = squared_errors.mean().item()
#     rmse = math.sqrt(mse)
    
#     # Print metrics for entire test set
#     print(f'\nTest Set Performance (all {len(all_outputs)} samples):')
#     print(f'Mean Absolute Error (MAE): {mae:.2f} years')
#     print(f'Mean Squared Error (MSE): {mse:.2f} years²')
#     print(f'Root Mean Squared Error (RMSE): {rmse:.2f} years')
    
#     # Show random subset of predictions for visualization
#     num_samples_to_show = min(20, len(all_outputs))  # Show fewer samples for readability
#     random_indices = torch.randperm(len(all_outputs))[:num_samples_to_show]
#     selected_outputs = all_outputs[random_indices]
#     selected_labels = all_labels[random_indices]
    
#     print(f'\nRandom sample of {num_samples_to_show} predictions:')
#     for i in range(len(selected_outputs)):
#         error = torch.abs(selected_outputs[i] - selected_labels[i]).item()
#         print(f'Predicted Age: {selected_outputs[i].item():.2f}, Actual Age: {selected_labels[i].item():.2f}, Error: {error:.2f}')


dataset = BrainAgeDataset(X_test, y_test)
test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)  # No need to shuffle for evaluation

age_predictor.eval()  # Set model to evaluation mode (no gradients)
test_loss = 0.0
num_total = 0
total_absolute_error = 0.0
all_outputs = []
all_labels = []

with torch.no_grad():  # Disable gradient calculation for testing
    for inputs, labels in test_loader:
        # Move data to the same device as the model and add channel dimension
        inputs = inputs.to(device).unsqueeze(1)
        labels = labels.to(device).float()
        
        with torch.amp.autocast(device_type='cuda'):
            outputs = age_predictor(inputs)  # Forward pass
            outputs = outputs.view(-1)  # Flatten outputs
            
            # Store predictions and labels
            all_outputs.append(outputs)
            all_labels.append(labels)
            
            # Print predictions for this batch
            print(f'Predicted Age: {outputs.squeeze().cpu().numpy()}, Actual Age: {labels.cpu().numpy()}')
            
            # Calculate loss
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            num_total += labels.size(0)
            
            # Calculate absolute error
            absolute_error = torch.abs(outputs - labels)
            total_absolute_error += absolute_error.sum().item()

# Stack all predictions and labels
all_outputs = torch.cat(all_outputs, dim=0)
all_labels = torch.cat(all_labels, dim=0)

# Calculate metrics
avg_test_loss = test_loss / len(test_loader)
mae = total_absolute_error / num_total
squared_errors = (all_outputs - all_labels) ** 2
mse = squared_errors.mean().item()
rmse = math.sqrt(mse)

print(f"\nTest Set Performance (all {num_total} batches):")
print(f"Test Loss: {avg_test_loss:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.2f} years")
print(f"Mean Squared Error (MSE): {mse:.2f} years²")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} years")