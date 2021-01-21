import torch
import torch.nn as nn
import torch.nn.functional as F

# InfoGAN Generator
class Generator(nn.Module):
    def __init__(self, dataset='MNIST', z_dim=62, cc_dim=2, dc_dim=1):
        # noise variable = z_dim (MNIST인 경우,  62)
        # continous latent codes = cc_dim (MNIST의 경우, 2)
        # ten-dimensional categorical code = dc_dim (MNIST의 경우, 1 => 1 * 10)
        # concatenated dimension = (MNIST의 경우 62 + 2 + 1 * 10)
        super(Generator, self).__init__()

        self.dataset = dataset

        if self.dataset == 'MNIST':
            self.dec_fc = nn.Sequential(
                nn.Linear(z_dim + cc_dim + (dc_dim * 10), 1024),
                nn.BatchNorm2d(1024),
                nn.ReLU(),

                nn.Linear(1024, 128 * 7 * 7, bias=False),
                nn.BatchNorm2d(128 * 7 * 7),
                nn.ReLU()
            )

            # [-1, 128, 7, 7] => [-1, 64, 14, 14]

            self.dec_conv = nn.Sequential(
                # [-1, 128, 7, 7] => [-1, 64, 14, 14]
                nn.ConvTranspose2d(128, 64, 4, 2, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(),

                nn.ConvTranspose2d(64, 1, 4, 2, 1),
                nn.Tanh()
            )
            

    def forward(self, x):
        if self.dataset == 'MNIST':
            # [-1, z]
            x = self.dec_fc(x)

            # [-1, 128 * 7 * 7] => [-1, 128, 7, 7]
            x = x.view(-1, 128, 7, 7)
            x = self.dec_conv(x)
        
        return x


# InfoGAN Discriminator
class Discriminator(nn.Module):
    def __init__(self, dataset='MNIST', cc_dim=2, dc_dim=1):
        # continous latent codes = cc_dim (MNIST의 경우, 2)
        # ten-dimensional categorical code = dc_dim (MNIST의 경우, 1 => 1 * 10)
        super(Discriminator, self).__init__()
        
        self.dataset = dataset
        self.cc_dim = cc_dim
        self.dc_dim = dc_dim
        
        if self.dataset == 'MNIST':
            self.enc_conv = nn.Sequential(
                # 28 x 28 => 14 x 14
                nn.Conv2d(1, 64, 4, 2, 1),
                nn.LeakyReLU(0.1, inplace=True),

                # 14 x 14 => 7 x 7
                nn.Conv2d(64, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.1, inplace=True)
            )

            self.enc_fc = nn.Sequential(
                nn.Linear(128 * 7 * 7, 128),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Linear(128, 1 + cc_dim, (dc_dim * 10))
            )

    def forward(self, x):
        if self.dataset == 'MNIST':
            # 28 x 28 => 128 * 7 * 7
            x = self.enc_conv(x)
            x = x.view(-1, 128 * 7 * 7)

            # 128 * 7 * 7 => 1 + cc_dim + dc_dim
            x = self.enc_fc(x)

        # Discrimination output
        x[:, 0] = F.sigmoid(x[:, 0].clone())

        # Continouse Code output = Value Itself
        # Discrete Code output (Class -> Softmax)
        x[:, self.cc_dim + 1 : (self.cc_dim + 1 + self.dc_dim)] = F.softmax(x[:, self.cc_dim + 1 : (self.cc_dim + 1 + self.dc_dim)].clone())

        return x