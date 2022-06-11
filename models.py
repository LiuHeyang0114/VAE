import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size2, z_dim):
        super(VAE, self).__init__()

        # encoder part
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size2)
        self.fc_mu = nn.Linear(hidden_size2, z_dim)
        self.fc_var = nn.Linear(hidden_size2, z_dim)
        # decoder part
        self.decoder_layer=nn.Sequential(
            nn.Linear(z_dim, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )

    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        mu = self.fc_mu(h)
        var = self.fc_var(h)
        return mu, var

    def sampling(self, mu, var):
        std = torch.exp(0.5 * var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def decoder(self, z):
        return self.decoder_layer(z)

    def forward(self, x):
        mu, var = self.encoder(x.view(-1, 784))
        z = self.sampling(mu, var)
        return self.decoder(z), mu, var


def loss_function(recon_x, x, mu, var):
    Binary_CE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KL_Distance = -0.5 * torch.sum(1 + var - mu.pow(2) - var.exp())
    loss = Binary_CE + KL_Distance
    return loss
