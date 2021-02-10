import torch
import torch.nn as nn

# Pix2Pix Generator
class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, nker=64, norm='bnorm'):
        super(Generator, self).__init__()

        # encoder
        self.enc1 = CBR2d(in_channels, 1 * nker, kernel_size=4, stride=2, padding=1, norm=None, relu=0.2)
        self.enc2 = CBR2d(1 * nker, 2 * nker, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.2)
        self.enc3 = CBR2d(2 * nker, 4 * nker, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.2)
        self.enc4 = CBR2d(4 * nker, 8 * nker, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.2)
        self.enc5 = CBR2d(8 * nker, 8 * nker, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.2)
        self.enc6 = CBR2d(8 * nker, 8 * nker, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.2)
        self.enc7 = CBR2d(8 * nker, 8 * nker, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.2)
        self.enc8 = CBR2d(8 * nker, 8 * nker, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.2)
        
        # U-Net decoder
        self.dec1 = DECBRD2d(8 * nker, 8 * nker, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.0, drop=0.5)
        # skip connection
        self.dec2 = DECBRD2d(2 * 8 * nker, 8 * nker, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.0, drop=0.5)
        self.dec3 = DECBRD2d(2 * 8 * nker, 8 * nker, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.0, drop=0.5)
        self.dec4 = DECBRD2d(2 * 8 * nker, 8 * nker, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.0, drop=None)
        self.dec5 = DECBRD2d(2 * 8 * nker, 4 * nker, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.0, drop=None)
        self.dec6 = DECBRD2d(2 * 4 * nker, 2 * nker, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.0, drop=None)
        self.dec7 = DECBRD2d(2 * 2 * nker, 1 * nker, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.0, drop=None)
        self.dec8 = DECBRD2d(2 * 1 * nker, out_channels, kernel_size=4, stride=2, padding=1, norm=None, relu=None, drop=None)

        self.tanh = nn.Tanh()

    def forward(self, x):
        # encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        enc6 = self.enc6(enc5)
        enc7 = self.enc7(enc6)
        enc8 = self.enc8(enc7)
        
        # decoder
        dec1 = self.dec1(enc8)
        
        cat2 = torch.cat((dec1, enc7), dim=1)
        dec2 = self.dec2(cat2)

        cat3 = torch.cat((dec2, enc6), dim=1)
        dec3 = self.dec3(cat3)

        cat4 = torch.cat((dec3, enc5), dim=1)
        dec4 = self.dec4(cat4)

        cat5 = torch.cat((dec4, enc4), dim=1)
        dec5 = self.dec5(cat5)

        cat6 = torch.cat((dec5, enc3), dim=1)
        dec6 = self.dec6(cat6)

        cat7 = torch.cat((dec6, enc2), dim=1)
        dec7 = self.dec7(cat7)

        cat8 = torch.cat((dec7, enc1), dim=1)
        dec8 = self.dec8(cat8)

        x = self.tanh(dec8)

        return x

# Pix2Pix Discriminator
class Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels, nker=64, norm='bnorm'):
        super(Discriminator, self).__init__()

        self.enc1 = CBR2d(1 * in_channels, 1 * nker, kernel_size=4, stride=2, padding=1, norm=None, relu=0.2, bias=False)
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

        return x

class DECBRD2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, norm='bnorm', relu=0.0, drop=0.5):
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

        if not drop is None:
            layers += [nn.Dropout2d(drop)]

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