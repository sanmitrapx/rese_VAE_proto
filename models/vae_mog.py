import numpy as np
from typing import Any
import torch
import torch.nn as nn
import torch.distributions as td

from models.encoder import Encoder
from models.decoder import Decoder, ReshapeRightmost

class MoGPrior(nn.Module):
    """
    A MoG prior on a K-dimensional code:
        p(z|w_1...w_C, u_1...u_C, s_1...s_C)
         = \sum_c w_c prod_k Normal(z[k]|u[c], s[c]^2)
    """

    def __init__(self, 
                z_dim: int, 
                num_components: int):
        """
        Args:
            z_dim: size of latent variable
            num_components: number of Gaussians to be mixed
        """
        super().__init__()
        outcome_shape = (z_dim,)
        self.outcome_shape = outcome_shape
        # [C]
        self.logits = nn.Parameter(\
            torch.rand(num_components, \
                       requires_grad=True), \
                        requires_grad=True)
        # (C,) + outcome_shape
        shape = (num_components,) + self.outcome_shape
        self.locs = nn.Parameter(\
            torch.rand(shape, requires_grad=True), \
            requires_grad=True)
        self.scales = nn.Parameter(\
            1 + torch.rand(shape, requires_grad=True), \
            requires_grad=True)
        self.num_components = num_components

    def forward(self, batch_shape: torch.Size) -> td.distribution.Distribution:

        shape = batch_shape + \
                (self.num_components,) + \
                self.outcome_shape
    
        # A draw from independent would return [B, C] 
        # draws of K-dimensional outcomes
        comps = td.Independent(\
                td.Normal(loc=self.locs.expand(shape), 
                        scale=self.scales.expand(shape)), 
                        len(self.outcome_shape))
        pc = td.Categorical(\
            logits=self.logits.expand(batch_shape + \
            (self.num_components,)))
        
        return td.MixtureSameFamily(pc, comps)
    
class BernoulliLikelihood(nn.Module):
    """
    The likelihood density p(X|Z)

    """
    def __init__(self, 
                z_dim: int,
                hidden_dim: int,
                ):
        super().__init__()
        """
        Args:
            z: latent code size
            hidden_dim: output size of the decoder MLP 
        """
        self.decoder = Decoder(z_dim = z_dim,
                               hidden_dim = hidden_dim,
                               )
        self.outcome_shape = (self.decoder.num_channels, 
                              self.decoder.width, 
                              self.decoder.height)
        

    def ind(self, z: torch.Tensor)->td.distribution.Distribution:
        h = self.decoder(z)
        # identify each z sample with a single logit (per pixel per channel) 
        h = torch.einsum('ijknijk->nijk', h)
        return td.Independent(\
            td.Bernoulli(logits=h), \
            len(self.outcome_shape))
    
    def forward(self, z: torch.Tensor)->td.distribution.Distribution:
        h = self.decoder(z)
        return td.Independent(\
            td.Bernoulli(logits=h), \
            len(self.outcome_shape))
    

class Guide(nn.Module):
    """
    The variational approximation q(z|x), 
        q(z|x):=N(z|lambda_mu, lambda_sigma), where
        lambda_mu, lambda_sigma = Encoder(x)
    """

    def __init__(self, 
                z_dim: int, 
                hidden_dim: int,
                ):
        """
        Args:
            z: latent code size
            hidden_dim: output size of the encoder MLP 

        """
        super().__init__()
        outcome_shape = (z_dim,)
        self.outcome_shape = outcome_shape   

        self.encoder = Encoder(z_dim=z_dim,
                               hidden_dim=hidden_dim)

    def forward(self, x: torch.Tensor)->td.Distribution:
        lambda_mu, lambda_sigma = self.encoder(x)
        return td.Independent(\
            td.Normal(loc=lambda_mu, \
                      scale=lambda_sigma), 
                      len(self.outcome_shape))
    
class MoGVAE(nn.Module):
    """
    A wrapper class to combine the joint distribution 
    p(x,z)=p(x|z)p(z), a likelihood parameterised 
    by a decoder MLP, a learned Mixture of Gaussian prior, and
    a variational approximation q(z|x), a guide,
    parameterised by an encoder MLP resp.

    This is a VAE with a Mixture of  Gaussian prior:
    p(z):=  p(z|w_1...w_C, u_1...u_C, s_1...s_C)
         = \sum_c w_c prod_k Normal(z[k]|u[c], s[c]^2). 
    """

    def __init__(self, prior: nn.Module, 
                 likelihood: nn.Module, 
                 guide: nn.Module, 
                 device: Any = None):
        """
        Args:
            prior: nn.Module for the density p(z)
            likelihood: nn.Module for the density p(x|z)
            guide: nn.Module for the density q(z|x) 

        """
        super().__init__()
        self.prior = prior
        self.likelihood = likelihood
        self.guide = guide
        self.device = device
        self.to(device)

    def reconstruct_img(self, x: torch.Tensor)->torch.Tensor:
        """
        A helper function for reconstructing images,
        by sampling from the posterior predicitve
        distribution x'~p(x'|x)

        Args:
            x: input image
        """
        qz = self.guide(x)
        px_z = self.likelihood(qz.sample())
        return px_z.sample()

    def forward(self, x: torch.Tensor, sample_size: int=1)->torch.Tensor:
        """
        Args:
            x: batch of images
            sample_size: if 1 or more, we use multiple samples
                sample_size controls a sequential computation (a for loop)

        Returns:
            ELBO loss
        """
        outcome_shape = self.likelihood.outcome_shape
        obs_dims = len(outcome_shape)
        batch_shape = x.shape[:-obs_dims]

        qz = self.guide(x)
        pz = self.prior(batch_shape)

        log_p_x_z = torch.zeros(1, device=self.device)
        log_p_z = torch.zeros(1, device=self.device)
        log_q_z_x = torch.zeros(1, device=self.device)

        for _ in range(sample_size):

            # Obtain a sample (independent per pixel per channel)
            z = qz.sample(sample_shape=outcome_shape)
            # use the independent z per pixel per channel likelihood
            px_z = self.likelihood.ind(z)

            # Compute all three relevant densities:
            # p(x|z)
            log_p_x_z += torch.mean(px_z.log_prob(x))
            # q(z|x,lambda)
            log_q_z_x += torch.mean(qz.log_prob(z))
            # p(z)
            log_p_z += torch.mean(pz.log_prob(z))

        # Compute the sample mean for the different terms
        log_p_x_z = log_p_x_z / sample_size
        log_p_z = log_p_z / sample_size
        log_q_z_x = log_q_z_x / sample_size

        expected_loglikelihood = log_p_x_z
        try:  # not every design admits tractable KL
            kl_q_p = td.kl.kl_divergence(qz, pz)
        except NotImplementedError:
            kl_q_p = log_q_z_x - log_p_z

        elbo = expected_loglikelihood - kl_q_p

        loss = -elbo
        return loss