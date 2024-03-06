import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
import torch.optim as opt


from data.data_loader import setup_data_loaders
from models.vae_mog import MoGPrior, BernoulliLikelihood, Guide, MoGVAE


class OptCollection:
    """
    A wrapper class to update the parameters
    of the likelihood, prior and guide one at a time
    """

    def __init__(self, prior, likelihood, guide):
        self.prior = prior
        self.likelihood = likelihood
        self.guide = guide

    def zero_grad(self):
        self.prior.zero_grad()
        self.likelihood.zero_grad()
        self.guide.zero_grad()

    def step(self):
        self.prior.step()
        self.likelihood.step()
        self.guide.step()


def evaluate(vae, val_loader, device=None):
    """
    a function To carry out evaluation for one epoch

    Args:

        vae: A nn.Module object that wraps the model and guide
        val_loader: Torch's DataLoader for the
                    validation/test set
        device: Torch device

    Returns:
        total_epoch_loss_val: average validation loss
                            over one epoch
    """
    val_loss = 0.0
    with torch.no_grad():
        for x, _ in val_loader:
            x = x.to(device)
            val_loss += vae(x)
    normalizer_val = len(val_loader.dataset)
    total_epoch_loss_val = val_loss / normalizer_val
    return total_epoch_loss_val


def train_vae(params, device):

    batch_size = params["batch_size"]
    z_dim = params["z_dim"]
    num_comps = params["num_comps"]
    hidden_dim = params["hidden_dim"]
    NUM_EPOCHS = params["NUM_EPOCHS"]
    VAL_FREQUENCY = params["VAL_FREQUENCY"]

    train_loader, val_loader, test_loader = setup_data_loaders(batch_size=batch_size)

    mog_prior = MoGPrior(z_dim, num_comps)
    Bern_likelihood = BernoulliLikelihood(z_dim=z_dim, hidden_dim=hidden_dim)
    mog_guide = Guide(z_dim=z_dim, hidden_dim=hidden_dim)
    vae = MoGVAE(
        prior=mog_prior, likelihood=Bern_likelihood, guide=mog_guide, device=device
    )
    print("Running MixOfGaussian VAE")

    optimizer = OptCollection(
        opt.Adam(vae.likelihood.parameters(), lr=1e-4, weight_decay=1e-5),
        opt.Adam(vae.prior.parameters(), lr=1e-3),
        opt.Adam(vae.guide.parameters(), lr=1e-4, weight_decay=1e-5),
    )

    train_elbo = []
    val_elbo = []

    # training loop
    for epoch in range(NUM_EPOCHS):
        train_loss = 0.0
        for x, _ in train_loader:
            vae.train()
            optimizer.zero_grad()
            x = x.to(device)
            loss = vae(x)
            loss.backward()

            optimizer.step()
            train_loss += loss
        normalizer_train = len(train_loader.dataset)
        total_epoch_loss_train = train_loss / normalizer_train
        train_elbo.append(total_epoch_loss_train)
        print(
            "[epoch %03d]  average training loss: %.4f"
            % (epoch, total_epoch_loss_train)
        )

        if epoch % VAL_FREQUENCY == 0:
            # report test diagnostics
            vae.eval()
            total_epoch_loss_val = evaluate(vae, val_loader, device)
            val_elbo.append(total_epoch_loss_val)
            print(
                "[epoch %03d] average validation loss: %.4f"
                % (epoch, total_epoch_loss_val)
            )
            with torch.no_grad():
                # plot samples from the posterior predictives on validation data
                for images, y in val_loader:

                    rec_img = vae.reconstruct_img(images.to(device)).cpu()

                    plt.figure(figsize=(20, 10))
                    plt.axis("off")
                    plt.imshow(make_grid(rec_img, nrow=32).permute((1, 2, 0)))
                    plt.title("Conditional, p(z|x) samples at [epoch %03d]" % (epoch))
                    plt.savefig("./results/epoch_" + str(epoch) + ".png")
                    break


if __name__ == "__main__":

    device = torch.device("mps")
    params = {
        "batch_size": 256,
        "z_dim": 50,
        "hidden_dim": 512,
        "num_comps": 5,
        "NUM_EPOCHS": 25,
        "VAL_FREQUENCY": 5,
    }

    train_vae(params=params, device=device)
