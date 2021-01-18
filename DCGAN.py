"""
Architecture guidelines for stable Deep Convolutional GANs
- Replace any pooling layers with convolutions (discriminator) and fractional-strided convolutions (generator).
- Use batchnorm in both the generator and the discriminator.
- Remove fully connected hidden layers for deeper  architectures.
- Use ReLU activation in generator for all layers except for the output, which uses Tanh.
- Use LeakyReLU activation in the discriminator for all layers.
"""
import torch
import torch.nn as nn

class DCGAN(nn.Module):
    def __init__(self, in_channels, out_channels, nker=64, norm='bnorm'):
        super(DCGAN, self).__init__()

        # Linear vector (100,)과 matrix (100, 1, 1)의 정보량은 동일
        self.dec1 = DECBR2d(1 * in_channels, 8 * nker, kernel_size=4, stride=1, padding=0, norm=norm, relu=0.0, bias=False)
        self.dec2 = DECBR2d(8 * nker, 4 * nker, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.0, bias=False)
        self.dec3 = DECBR2d(4 * nker, 2 * nker, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.0, bias=False)
        self.dec4 = DECBR2d(2 * nker, nker, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.0, bias=False)
        self.dec5 = DECBR2d(nker, out_channels, kernel_size=4, stride=2, padding=1, norm=None, relu=None, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        x = self.dec5(x)

        x = self.tanh(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels, nker=64, norm='bnorm'):
        super(Discriminator, self).__init__()

        self.enc1 = CBR2d(1 * in_channels, 1 * nker, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.2, bias=False)
        self.enc2 = CBR2d(1 * nker, 2 * nker, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.2, bias=False)
        self.enc3 = CBR2d(2 * nker, 4 * nker, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.2, bias=False)
        self.env4 = CBR2d(4 * nker, 8 * nker, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.2, bias=False)
        self.env5 = CBR2d(8 * nker, out_channels, kernel_size=4, stride=2, padding=1, norm=None, relu=None, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc5(x)
        
        x = self.sig(x)

        return x

class DECBR2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, norm='bnorm', relu=0.0):
        super().__init__()

        layers = []
        layers += [nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]

        if not norm is None:
            if norm == 'bnorm':
                layers += [nn.BatchNorm2d(num_features=out_channels)]
            elif norm == 'inorm':
                layers += [nn.InstanceNorm2d(num_features=out_channels)]

        if not relu is None and relu >= 0.0:
            layers += [nn.ReLU() if relu == 0 else nn.LeakyReLU(relu)]

        self.cbr = nn.Sequential(*layers)

    def forward(self, x):
        return self.cbr(x)


class CBR2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, norm='bnorm', relu=0.0):
        super().__init__()

        layers = []
        layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]

        if not norm is None:
            if norm == 'bnorm':
                layers += [nn.BatchNorm2d(num_features=out_channels)]
            elif norm == 'inorm':
                layers += [nn.InstanceNorm2d(num_features=out_channels)]

        if not relu is None and relu >= 0.0:
            layers += [nn.ReLU() if relu == 0 else nn.LeakyReLU(relu)]

        self.cbr = nn.Sequential(*layers)

    def forward(self, x):
        return self.cbr(x)