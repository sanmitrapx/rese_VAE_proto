import numpy as np
import torch
import torch.nn as nn

class ReshapeRightmost(nn.Module):
    """
    Helper layer to reshape the rightmost dimension of a tensor.

    This can be used as a component of nn.Sequential.
    """
    def __init__(self, shape):
        """
        Args:
            shape: desired rightmost shape
        """        
        super().__init__()
        self._shape = shape
    def forward(self, input):
        # reshapes the last dimension into self.shape
        return input.reshape(input.shape[:-1] + self._shape)
        
class Decoder(nn.Module):
    """
    Map the latent variable z to a tensor with shape 
    [Batch, num_channels, width, height]
    using a FFNN with 2 hidden layers.
    """
    def __init__(self, z_dim: int, 
            hidden_dim: int, 
            num_channels: int=1, 
            width: int=64, 
            height: int=64, 
            p_drop: int=0.1):
        super().__init__()
        """
        Args:
            z_dim: size of latent variable
            hidden_dim: we first map from z_dim to hidden_dim and
                then use feed forward NNs to map it to [num_channels, width, height]
            num_channels: number of channels in the output image
            width: image shape
            height: image shape
            p_drop: dropout rate before linear layers
        """
        self.decoder = nn.Sequential(
        nn.Dropout(p_drop),
        nn.Linear(z_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(p_drop),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(p_drop),
        nn.Linear(hidden_dim, num_channels * width * height),
        ReshapeRightmost((num_channels, width, height)),
        )

        self.width = width
        self.height = height
        self.num_channels = num_channels

    def forward(self, z: torch.Tensor)-> torch.Tensor:
        """
        Define the forward computation on the latent z and
        return the parameter for the data (likelihood) distribution
        each is of size [batch_size, num_channels, width, height] 

        Args:
            z: latent code

        Returns:
            logits_img: decoder output parameterising 
                        the Bernouli liklihood

        """

        logits_img = self.decoder(z)
        return logits_img