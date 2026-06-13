# Feature extraction for custom CNN - Dental Paper

import torch
import torch.nn as nn

class FeatureExtraction(nn.Module):
    """
    FeatureExtraction Module for Dental Radiograph Analysis.
    
    This module extracts hierarchical features using a series of Convolutional layers.
    Input: Single-channel (grayscale) images of size 128x128.
    Output: High-dimensional feature maps of size 16x16 with 256 channels.
    """
    def __init__(self):
        super(FeatureExtraction, self).__init__()

        self.feature_extraction = nn.Sequential(
            # Block 1: Initial feature extraction (edges and basic textures)
            # Input shape: [Batch, 1, 128, 128]
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Downsampling to 64x64
            nn.Dropout(p=0.5),

            # Block 2: Intermediate feature extraction
            # Input shape: [Batch, 64, 64, 64]
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Downsampling to 32x32
            nn.Dropout(p=0.5),

            # Block 3: Complex deep feature extraction
            # Input shape: [Batch, 128, 32, 32]
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Final downsampling to 16x16
            nn.Dropout(p=0.5)
            
            # Final Feature Map output: [Batch, 256, 16, 16]
        )

    def forward(self, x):
        """
        Forward pass for extracting feature maps.
        """
        x = self.feature_extraction(x)
        return x


# # TESTing
# # Dimension Verification
# if __name__ == "__main__":
#     # Initialize model
#     model = FeatureExtraction()
    
#     # Create a dummy input (Batch size = 1, Channels = 1, Height = 128, Width = 128)
#     sample_input = torch.randn(1, 1, 128, 128)
    
#     # Forward pass
#     output = model(sample_input)
    
#     print(f"Input Shape: {sample_input.shape}")
#     print(f"Output Shape (Spatial 16x16): {output.shape}")
#     # Expected Output: torch.Size([1, 256, 16, 16])
