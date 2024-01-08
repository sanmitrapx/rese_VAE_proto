from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist

from models.encoder import Encoder
from models.decoder import Decoder

class StandardVAE(nn.Module):
    """
    A wrapper class to combine the joint distribution 
    p(x,z)=p(x|z)p(z), a model, and
    a variational approximation q(z|x), 
    a guide, parameterised by a decoder and an encoder MLP
    resp.

    This is a VAE with a hierarchical Gaussian prior:
    p(z):=  N(z|z_mu,z_sigma), where z_mu=0, z_sigma=1 .
    """
    def __init__(self, z_dim: int=50, 
                hidden_dim: int=512,
                device: torch.device=None):
        super().__init__()
        """
        Args:
            z_dim: size of latent variable
            hidden_dim: hidden layer size for MLPs
            device: 
        """    
        self.encoder = Encoder(z_dim, hidden_dim)
        self.decoder = Decoder(z_dim, hidden_dim)

        self.to(device)
        self.z_dim = z_dim


    def model(self, x: torch.Tensor)->torch.Tensor:
        """
        The joint distribution p(x|z)p(z), as
        a pyro model, where p(x|z):=Bern(x|eta),
        and eta = Decoder(z)
        
        Args:
           x: batch of input images 
        """   
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            # prior p(z)
            z_mu = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
            z_sigma = x.new_ones(torch.Size((x.shape[0], self.z_dim)))
            z = pyro.sample("latent", \
                            dist.Normal(z_mu, z_sigma).to_event(1))

            logits = self.decoder(z)

            # likelihood p(x|z)
            img = pyro.sample("obs",\
                             dist.Bernoulli(logits).to_event(3), 
                             obs=x)
            return logits
   

    def guide(self, x: torch.Tensor):
        """
        The variational approximation q(z|x), as
        a pyro guide.
        q(z|x):=N(z|lambda_mu, lambda_sigma), where
        lambda_mu, lambda_sigma = Encoder(x)

        Args:
           x: batch of input images 
        """           
        pyro.module("encoder", self.encoder)
        
        lambda_mu, lambda_sigma = self.encoder(x)

        with pyro.plate("data", x.shape[0]):

            # sample the latent variable from z~q(z|x)
            pyro.sample("latent", dist.Normal(\
                lambda_mu, lambda_sigma).to_event(1))


    def reconstruct_img(self, 
                        x: torch.Tensor)-> \
                        Tuple[torch.Tensor, torch.Tensor]:
        """
        A helper function for reconstructing images,
        by sampling from the posterior predicitve
        distribution x'~p(x'|x)
        
        Args:
           x: batch of input images 

        Returns:
            rec_logits: decoder output parameterising 
                        the Bernouli liklihood
            z: posterior saple of the latent code
        """

        lambda_mu, lambda_sigma = self.encoder(x)
        z = dist.Normal(lambda_mu, lambda_sigma).sample()
        rec_logits = self.decoder(z)
        return rec_logits, z