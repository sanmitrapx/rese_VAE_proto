import sys
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import pyro.distributions as dist

from data.data_loader import setup_data_loaders
from models.vae_standard_pyro import StandardVAE as sVAE
from models.vae_hierarch_pyro import HierarchVAE as hVAE

def train(svi: SVI, 
          train_loader: DataLoader,
          device: torch.device=None)->torch.Tensor:
    """
    Use Pyro's ELBO loss stepper class svi
    to carry out training for one epoch
    
    Args:
        svi: Pyro object that wraps the model and guide
        train_loader: Torch's DataLoader for the training 
                    set
        device: Torch device

    Returns:
        total_epoch_loss_val: average training 
                            loss over one epoch        
    """      
    epoch_loss = 0.
    for x, _ in train_loader:
        x = x.to(device)
        epoch_loss += svi.step(x)

    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = epoch_loss / normalizer_train
    return total_epoch_loss_train
    
def evaluate(svi: SVI, 
             val_loader: DataLoader, 
             device: torch.device=None)->torch.Tensor:
    """
    Use Pyro's ELBO loss stepper class svi
    to carry out evaluation for one epoch
    
    Args:

        svi: Pyro object that wraps the model and guide
        train_loader: Torch's DataLoader for the 
        validation/test set
        device: Torch device

    Returns:
        total_epoch_loss_val: average validation loss 
                            over one epoch
    """      
    val_loss = 0.
    for x, _ in val_loader:
        x = x.to(device)
        val_loss += svi.evaluate_loss(x)
    normalizer_val = len(val_loader.dataset)
    total_epoch_loss_val = val_loss / normalizer_val
    return total_epoch_loss_val

def train_vae(params, device):
    type_of_VAE = params["type_of_VAE"]
    batch_size = params["batch_size"]
    z_dim=params["z_dim"]
    hidden_dim=params["hidden_dim"]
    LEARNING_RATE = params["learning_rate"]
    NUM_EPOCHS = params["NUM_EPOCHS"]
    VAL_FREQUENCY = params["VAL_FREQUENCY"]
    
    train_loader, val_loader, test_loader\
    = setup_data_loaders(batch_size=batch_size)

    pyro.distributions.enable_validation(False)
    pyro.set_rng_seed(0)
    pyro.clear_param_store()

    if type_of_VAE=="h":
        print("Running hierarchical VAE")
        vae = hVAE(z_dim=z_dim,
                hidden_dim=hidden_dim, 
                device=device) 
    else:
        print("Running standard VAE")
        vae = sVAE(z_dim=z_dim,
                hidden_dim=hidden_dim, 
                device=device)      
    
    adam_args = {"lr": LEARNING_RATE}
    optimizer = Adam(adam_args)

    svi = SVI(vae.model,
         vae.guide, 
         optimizer, 
         loss=Trace_ELBO())
    
    train_elbo = []
    val_elbo = []
    # training loop
    for epoch in range(NUM_EPOCHS):
        vae.train()
        total_epoch_loss_train = train(svi, train_loader, device)
        train_elbo.append(-total_epoch_loss_train)
        print("[epoch %03d]  average training loss: %.4f" 
              % (epoch, total_epoch_loss_train))

        if epoch % VAL_FREQUENCY == 0:
            vae.eval()
            # report test diagnostics
            total_epoch_loss_val = evaluate(svi, val_loader, device)
            val_elbo.append(-total_epoch_loss_val)
            print("[epoch %03d] average validation loss: %.4f" 
                  % (epoch, total_epoch_loss_val))
            with torch.no_grad():
                # plot samples from the posterior predictives on validation data
                for images, y in val_loader:
                    rec_loc = vae.reconstruct_img(
                        images.to(device))[0]
                    rec_img = dist.Bernoulli(
                        rec_loc).to_event(3).sample().cpu()
                    
                    plt.figure(figsize=(20,10))
                    plt.axis('off')
                    plt.imshow(make_grid(rec_img, nrow=32).permute((1, 2, 0)))
                    plt.title("Conditional, p(z|x) samples at [epoch %03d]" 
                              %(epoch))
                    plt.savefig('./results/figures/epoch_'+str(epoch)+'.png')
                    break


if __name__ == '__main__':

    device = torch.device("mps")
    if len(sys.argv)>1:
        arg = sys.argv[1]
    else:
        arg = "s"
    params = {"type_of_VAE":arg,
    "batch_size": 256,
    "z_dim": 50,
    "hidden_dim":512,
    "learning_rate": 1.0e-3,
    "NUM_EPOCHS":25,
    "VAL_FREQUENCY":5,
    }
    
    train_vae(params=params, device=device)

    









