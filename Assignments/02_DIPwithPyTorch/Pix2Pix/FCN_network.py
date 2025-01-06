import torch
import torch.nn as nn

class FullyConvNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder (Convolutional Layers)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        
        # Additional encoder layers
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Decoder (Deconvolutional Layers)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Using Tanh to map output to [-1,1], common for image outputs
        )

    def forward(self, x):
        # Encoder forward pass
        x = self.conv1(x)   # [B, 8, ..., ...]
        x = self.conv2(x)   # [B, 16, ..., ...]
        x = self.conv3(x)   # [B, 32, ..., ...]
        x = self.conv4(x)   # [B, 64, ..., ...]
        
        # Decoder forward pass
        x = self.deconv1(x) # [B, 32, ..., ...]
        x = self.deconv2(x) # [B, 16, ..., ...]
        x = self.deconv3(x) # [B, 8,  ..., ...]
        x = self.deconv4(x) # [B, 3,  ..., ...]
        
        output = x
        return output
    