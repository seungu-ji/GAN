import torch
import torch.nn as nn

# CGAN Generator
# in_features = latent_dim + num_class
# out_features = int(np.prod(channels, img_size, img_size))
class Generator(nn.Module):
    def __init__(self,  latent_dim=100, num_class=10, nker=128, norm='bnorm'):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(num_class, num_class)

        # Linear vector (100,)과 matrix (100, 1, 1)의 정보량은 동일
        self.dec1 = LDBR2d(latent_dim + num_class, nker, dropout=False, norm=norm, relu=0.2)
        self.dec2 = LDBR2d(nker, 2 * nker, dropout=False, norm=norm, relu=0.2)
        self.dec3 = LDBR2d(2 * nker, 4 * nker, dropout=False, norm=norm, relu=0.2)
        self.dec4 = LDBR2d(4 * nker, 8 * nker, dropout=False, norm=norm, relu=0.2)
        self.dec5 = LDBR2d(8 * nker, (1 * 32 * 32), dropout=False, norm=None, relu=None)
        self.tanh = nn.Tanh()

    # y = label
    def forward(self, z, y):
        x = torch.cat((self.label_emb(y), z), -1)

        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        x = self.dec5(x)
        x = self.tanh(x)

        x = x.view(x.size(0), *(1, 32, 32))
        return x


# CGAN Discriminator
class Discriminator(nn.Module):
    def __init__(self,  latent_dim=100, num_class=10, nker=128, norm='bnorm'):
        super(Discriminator, self).__init__()

        self.label_emb = nn.Embedding(num_class, num_class)

        self.enc1 = LDBR2d(num_class + (1 * 32 * 32), 4 * nker, dropout=False, norm=None, relu=0.2)
        self.enc2 = LDBR2d(4 * nker, 4 * nker, dropout=True, norm=None, relu=0.2)
        self.enc3 = LDBR2d(4 * nker, 4 * nker, dropout=True, norm=None, relu=0.2)
        self.env4 = LDBR2d(4 * nker, 1, dropout=False, norm=None, relu=None)
        # self.sig = nn.Sigmoid()

    # y = label
    def forward(self, x, y):
        x = torch.cat((x.view(x.size(0), -1), self.label_emb(y)), -1)

        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        
        # x = self.sig(x)

        return x


class LDBR2d(nn.Module):
    def __init__(self, in_features, out_features, dropout=False, norm='bnorm', relu=0.0):
        super().__init__()

        layers = []
        layers += [nn.Linear(in_features=in_features, out_features=out_features)]

        if dropout == True:
            layers += [nn.Dropout(0.4)]

        if not norm is None:
            if norm == 'bnorm':
                layers += [nn.BatchNorm1d(num_features=out_features)]
            elif norm == 'inorm':
                layers += [nn.InstanceNorm1d(num_features=out_features)]

        if relu >= 0.0:
            layers += [nn.ReLU() if relu == 0 else nn.LeakyReLU(relu)]

        self.ldbr = nn.Sequential(*layers)

    def forward(self, x):
        return self.ldbr(x)