import torch
import torch.nn as nn

# SRGAN Generator = SRResNet
class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, nker=64, norm='bnorm', nblock=16):
        super(Generator, self).__init__()

        # Linear vector (100,)과 matrix (100, 1, 1)의 정보량은 동일
        self.enc = CBR2d(in_channels, nker, kernel_size=9, stride=1, padding=4, norm=None, relu=0.0, bias=True)
        
        resblocks = []

        for i in range(nblock):
            res += [ResBlock(nker, nker, kernel_size=3, stride=1, padding=1, bias=True, norm=norm, relu=0.0)]

        self.res = nn.Sequential(*resblocks)

        self.dec = CBR2d(nker, nker, kernel_size=3, stride=1, padding=1, bias=True, norm=norm, relu=None)

        ps1 = []
        ps1 += [nn.Conv2d(in_channels=nker, out_channels=4 * nker, kernel_size=3, stride=1, padding=1)]
        ps1 += [PixelShuffle(ry=2, rx=2)]
        ps1 += [nn.ReLU()]
        self.ps1 = nn.Sequential(*ps1)

        ps2 = []
        ps2 += [nn.Conv2d(in_channels=nker, out_channels=4 * nker, kernel_size=3, stride=1, padding=1)]
        ps2 += [PixelShuffle(ry=2, rx=2)]
        ps2 += [nn.ReLU()]

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
        

"""# DCGAN Discriminator
class Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels, nker=64, norm='bnorm'):
        super(Discriminator, self).__init__()

        self.enc1 = CBR2d(1 * in_channels, 1 * nker, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.2, bias=False)
        self.enc2 = CBR2d(1 * nker, 2 * nker, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.2, bias=False)
        self.enc3 = CBR2d(2 * nker, 4 * nker, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.2, bias=False)
        self.enc4 = CBR2d(4 * nker, 8 * nker, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.2, bias=False)
        self.enc5 = CBR2d(8 * nker, out_channels, kernel_size=4, stride=2, padding=1, norm=None, relu=None, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc5(x)
        
        x = self.sig(x)

        return x"""


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