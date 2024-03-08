from typing import Tuple
import numpy as np
import torch
import torch.nn as nn


class FlattenImage(nn.Module):
    def forward(self, input: torch.Tensor):
        return input.reshape(input.shape[:-3] + (-1,))


class Encoder(nn.Module):
    """
    Map the image of shape [Batch, num_channels, width, height]
    to a latent code (variable) z
    with shape [Batch, num_channels, width, height]
    using a FFNN with 2 hidden layers.
    """

    def __init__(
        self,
        z_dim: int,
        hidden_dim: int,
        num_channels: int = 1,
        width: int = 64,
        height: int = 64,
        p_drop: float = 0.1,
    ):
        super().__init__()
        """
        Args:
            z_dim: size of latent variable
            hidden_dim: we first map from z_dim
                        to hidden_dim and
                        then use feed forward 
                        NNs to map it to z_dim
            num_channels: number of channels in the input image
            width: image shape
            height: image shape
            p_drop: dropout rate before linear layers

        """
        self.encoder = nn.Sequential(
            FlattenImage(),
            nn.Dropout(p_drop),
            nn.Linear(num_channels * width * height, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )
        self.locs = nn.Sequential(nn.Dropout(p_drop), nn.Linear(hidden_dim, z_dim))
        self.scales = nn.Sequential(nn.Dropout(p_drop), nn.Linear(hidden_dim, z_dim))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: a batch of images

        Returns:
            lambda_mu: encoded posterior mean
            lambda_sigma: encoded posterior (positive)
                            square root covariance
        """

        hidden = self.encoder(x)

        lambda_mu = self.locs(hidden)
        lambda_sigma = torch.exp(self.scales(hidden))
        return lambda_mu, lambda_sigma
