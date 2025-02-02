import torch
import torch.nn as nn


class ResnetBlock(nn.Module):
    """Residual Block, with Instance Normalization"""

    def __init__(self, dim):
        # TODO
        self.block = None

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
        # TODO
        self.model = None

    def forward(self, x):
        return self.model(x)
