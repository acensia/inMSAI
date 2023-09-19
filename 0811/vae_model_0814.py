import torch.nn as nn
import torch

# 10 ~ 100
latent_dim = 20


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(),
            nn.Linear(400, 2 * latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, layer):
        std = torch.exp(0.5 * layer)
        epsilon = torch.randn_like(std)


        return mu + epsilon
    """
        reparameterize potential variable
        mu : mean value of potential variable
        layer : log variance from VAE encoder
    """


    def forward(self, x):
        x = x.view(-1, 784)
        mu_logval = self.encoder(x)
        mu = mu_logval[:, :latent_dim]
        logvar = mu_logval[:, latent_dim:]
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)

        return x_recon, mu, logvar