import torch
import torch.nn as nn


def adversarial_loss(D, real, fake):
    """
    Least Squares GAN (LSGAN) loss for a discriminator

    Args:
        D (nn.Module): discriminator network
        real (torch.Tensor): batch of real images from dataset
        fake (torch.Tensor): batch of generated (fake) images
    Returns:
        torch.Tensor: adversarial loss
    """
    real_loss = torch.mean((D(real) -1)**2)
    fake_loss = torch.mean((D(fake))**2)
    return (real_loss + fake_loss)


def cycle_consistency_loss(G, F, real_X, real_Y):
    """
    Cycle consistency loss
    makes sure the mapping to the other domain and back returns close to the original image

    Args:
        G (nn.Module): Generator mapping X → Y
        F (nn.Module): Generator mapping Y → X
        real_X (torch.Tensor): A batch of real images from domain X
        real_Y (torch.Tensor): A batch of real images from domain Y

    Returns:
        torch.Tensor: cycle consistency loss
    """
    x = F(G(real_X))
    y = G(F(real_Y))

    x_loss = nn.L1Loss(x, real_x)
    y_loss = nn.L1Loss(y, real_y)
    return x_loss + y_loss


def identity_loss(G, F, real_X, real_Y):
    """
    Computes identity loss
    make sure that an image from domain X mapped by F remains unchanged, and vice versa
    so enforces F(x_i) = x_i, G(y_i) = y_i

    Args:
        G (nn.Module): Generator mapping X → Y
        F (nn.Module): Generator mapping Y → X
        real_X (torch.Tensor): A batch of real images from domain X
        real_Y (torch.Tensor): A batch of real images from domain Y

    Returns:
        torch.Tensor: The identity loss value.
    """
    x = F(real_X)
    y = G(real_Y)

    x_loss = nn.L1Loss(x, real_x)
    y_loss = nn.L1Loss(y, real_y)
    return x_loss + y_loss


# COULD BE MORE LOSS FUNCTIONS
