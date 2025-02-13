import torch
import torch.nn as nn


class ResnetBlock(nn.Module):
    """Residual Block, with Instance Normalization"""
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        # Define the layers in the residual block
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),  # Reflective padding to preserve spatial dimensions
            nn.Conv2d(in_features, in_features, kernel_size=3),  # Convolution layer
            nn.InstanceNorm2d(in_features),  # Instance normalization
            nn.ReLU(inplace=True),  # ReLU activation
            nn.ReflectionPad2d(1),  # Reflective padding for the second conv layer
            nn.Conv2d(in_features, in_features, kernel_size=3),  # Second convolution layer
            nn.InstanceNorm2d(in_features)  # Instance normalization
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    """CycleGAN Generator"""

    def __init__(self, input_nc, output_nc, num_res_blocks=6):
        """
        Args:
            input_nc (int): number of channels (like 3 channels for rgb) of input image
            output_nc (int): number of channels of output image
            num_res_blocks (int): number of residual blocks (in paper says 6 for 128x128, 9 for 256x256 images)
        """
        super(Generator, self).__init__()

        # Initial Convolution Block
        out_features = 64  # Number of output features for the first layer
        model = [
            nn.ReflectionPad2d(3),  # Reflective padding for initial conv
            nn.Conv2d(input_nc, out_features, kernel_size=7),  # Initial convolution
            nn.InstanceNorm2d(out_features),  # Instance normalization
            nn.ReLU(inplace=True)  # ReLU activation
        ]
        in_features = out_features  # Update in_features for the next layer
        
        # Downsampling layers
        for _ in range(2):
            out_features *= 2  # Double the number of features
            model += [
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1),  # Downsampling conv
                nn.InstanceNorm2d(out_features),  # Instance normalization
                nn.ReLU(inplace=True)  # ReLU activation
            ]
            in_features = out_features  # Update in_features for next layer
        
        # Residual blocks
        for _ in range(num_res_blocks):
            model += [ResnetBlock(in_features)]  # Add residual blocks

        # Upsampling layers
        for _ in range(2):
            out_features //= 2  # Halve the number of features
            model += [
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # Upsampling layer
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=1, padding=1),  # Convolution after upsampling
                nn.InstanceNorm2d(out_features),  # Instance normalization
                nn.ReLU(inplace=True)  # ReLU activation
            ]
            in_features = out_features  # Update in_features for the next layer
            
        # Output layer
        model += [
            nn.ReflectionPad2d(3),  # Reflective padding for output
            nn.Conv2d(out_features, output_nc, kernel_size=7),  # Final convolution to output channels
            nn.Tanh()  # Tanh activation to scale output to [-1, 1]
        ]
        
        # Unpacking
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
