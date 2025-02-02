import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """CycleGAN Discriminator
    paper uses PatchGAN so we will too """

    def __init__(self, input_nc):
        """
        args:
            input_nc (int): number of channels (like 3 channels for rgb) of input image
        """

        self.model = None

    def forward(self, x):
        return self.model(x)
