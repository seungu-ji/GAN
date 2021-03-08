import torch
import torch.nn as nn

# SRGAN Generator = SRResNet
class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, nker=64, norm='bnorm', nblock=16):
        super(Generator, self).__init__()

        # Linear vector (100,)과 matrix (100, 1, 1)의 정보량은 동일
        self.enc = CBR2d(in_channels, nker, kernel_size=9, stride=1, padding=4, norm=None, relu='prelu', bias=True)
        
        resblocks = []

        for i in range(nblock):
            res += [ResBlock(nker, nker, kernel_size=3, stride=1, padding=1, bias=True, norm=norm, relu='prelu')]

        self.res = nn.Sequential(*resblocks)

        self.dec = CBR2d(nker, nker, kernel_size=3, stride=1, padding=1, bias=True, norm=norm, relu=None)

        ps1 = []
        ps1 += [nn.Conv2d(in_channels=nker, out_channels=4 * nker, kernel_size=3, stride=1, padding=1)]
        ps1 += [PixelShuffle(ry=2, rx=2)]
        ps1 += [nn.PReLU]
        self.ps1 = nn.Sequential(*ps1)

        ps2 = []
        ps2 += [nn.Conv2d(in_channels=nker, out_channels=4 * nker, kernel_size=3, stride=1, padding=1)]
        ps2 += [PixelShuffle(ry=2, rx=2)]
        ps2 += [nn.PReLU()]
        self.ps2 = nn.Sequential(*ps2)

        self.fc = CBR2d(nker, out_channels, kernel_size=9, stride=1, padding=4, bias=True, norm=None, relu=None)


    def forward(self, x):
        x = self.enc(x)
        identity = x

        x = self.res(x)

        x = self.dec(x)

        x = x + identity
        x = self.ps1(x)
        x = self.ps2(x)

        x = self.fc(x)

        return x
        

# SRGAN Discriminator
class Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels, nker=64, norm='bnorm'):
        super(Discriminator, self).__init__()

        discriminatorblocks = []

        for i, nker in enumerate([64, 128, 256, 512]):
            discriminatorblocks += [DiscriminatorBlock(in_channels, nker, first_block=(i == 0))]
            in_channels = nker

        self.discriminatorblocks = nn.Sequential(*discriminatorblocks)
        
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(nker, 2 * nker, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(2 * nker, out_channels, kernel_size=1)
        )

        self.sig = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size(0)

        x = self.discriminatorblocks(x)
        
        x = self.fc(x)

        return self.sig(x.view(batch_size))


class PixelShuffle(nn.Module):
    def __init__(self, ry=2, rx=2):
        super().__init__()
        self.ry = ry
        self.rx = rx

    def forward(self, x):
        ry = self.ry
        rx = self.rx

        [B, C, H, W] = list(x.shape)

        x = x.reshape(B, C // (ry * rx), ry, rx, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3)
        x = x.reshape(B, C // (ry * rx), H * ry, W * rx)

        return x

class PixelUnshuffle(nn.Module):
    def __init__(self, ry=2, rx=2):
        super().__init__()
        self.ry = ry
        self.rx = rx

    def forward(self, x):
        ry = self.ry
        rx = self.rx

        [B, C, H, W] = list(x.shape)

        x = x.reshape(B, C, H // ry, ry, W // rx, rx)
        x = x.permute(0, 1, 3, 5, 2, 4)
        x = x.reshape(B, C * (ry * rx), H // ry, W // rx)

        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, norm='bnorm', relu=0.0):
        super().__init__()

        layers = []

        layers += [CBR2d(in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, bias=bias, norm=norm, relu=relu)]
        
        layers += [CBR2d(in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, bias=bias, norm=norm, relu=None)]

        self.resblock = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.resblock(x)


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, norm='bnorm', relu=0.0, first_block=False):
        super().__init__()

        layers = []

        layers += [CBR2d(in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, bias=bias, norm=None, relu=None)]
        if not first_block:
                layers += [nn.BatchNorm2d(num_features=out_channels)]
        layers += [nn.LeakyReLU(0.2, inplace=True)]
        
        layers += [CBR2d(in_channels, out_channels, kernel_size=kernel_size,
                stride=2, padding=padding, bias=bias, norm=norm, relu=0.2)]

        self.discriminatorblock = nn.Sequential(*layers)

    def forward(self, x):
        return self.discriminatorblock(x)



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

        if relu == 'prelu':
            layers += [nn.PReLU]
        elif not relu is None and relu >= 0.0:
            layers += [nn.ReLU() if relu == 0 else nn.LeakyReLU(relu)]

        self.cbr = nn.Sequential(*layers)

    def forward(self, x):
        return self.cbr(x)