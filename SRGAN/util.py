import os
import numpy as np

import torch
import torch.nn as nn

from scipy.stats import poisson
from skimage.transform import rescale, resize

## Network grad setting
def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]

    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

## Network weights 초기화
def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__

        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)

            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


## Network save
def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'net': net.state_dict(), 'optim': optim.state_dict()}, '%s/model_epoch%d.pth' % (ckpt_dir, epoch))

def gan_save(ckpt_dir, netG, netD, optimG, optimD, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'netG': netG.state_dict(), 'netD': netD.state_dict(),
                'optimG': optimG.state_dict(), 'optimD': optimD.state_dict()},
               "%s/model_epoch%d.pth" % (ckpt_dir, epoch))


## Network load
def load(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_dir.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('%s%s' % (ckpt_dir, ckpt_lst[-1]))

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return net, optim, epoch

def gan_load(ckpt_dir, netG, netD, optimG, optimD):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return netG, netD, optimG, optimD, epoch

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('%s%s' % (ckpt_dir, ckpt_lst[-1]))

    netG.load_state_dict(dict_model['netG'])
    netD.load_state_dict(dict_model['netD'])
    optimG.load_state_dict(dict_model['optimG'])
    optimD.load_state_dict(dict_model['optimD'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return netG, netD, optimG, optimD, epoch

## Task
# Add sampling
def add_sampling(img, type='random', opts=None):
    size = img.shape

    if type == 'uniform':
        ds_y = opts[0].astype(np.int)
        ds_x = opts[1].astype(np.int)

        msk = np.zeros(img.shape)
        msk[::ds_y, ::ds_x, :] = 1

        dst = img * msk
    elif type == 'random':
        prob = opts[0]

        rnd = np.random.rand(size[0], size[1], size[2])
        msk = (rnd > prob).astype(np.float)

        dst = img * msk
    elif type == 'gaussian':
        x0 = opts[0]
        y0 = opts[1]
        sgmx = opts[2]
        sgmy = opts[3]

        a = opts[4]

        ly = np.linspace(-1, 1, size[0])
        lx = np.linspace(-1, 1, size[1])

        x, y = np.meshgrid(lx, ly)

        gaus = a * np.exp(-((x - x0)**2/(2*sgmx**2) + (y - y0)**2/(2*sgmy**2)))
        gaus = np.tile(gaus[:, :, np.newaxis], (1, 1, size[2]))
        rnd = np.random.rand(size[0], size[1], size[2])
        msk = (rnd < gaus).astype(np.float)

        dst = img * msk

    return dst

# Add noise
def add_noise(img, type='random', opts=None):
    size = img.shape

    if type == 'random':
        sgm = opts[0]

        noise = sgm / 255.0 * np.random.randn(size[0], size[1], size[2])

        dst = img + noise

    elif type == 'poisson':
        dst = poisson.rvs(255.0 * img) / 255.0
        noise = dst - img

    return dst

# Add blurring
def add_blur(img, type='bilinear', opts=None):
    if type == 'nearest':
        order = 0
    elif type == 'bilinear':
        order = 1
    elif type == 'biquadratic':
        order = 2
    elif type == 'bicubic':
        order = 3
    elif type == 'biquartic':
        order = 4
    elif type == 'biquintic':
        order = 5

    size = img.shape
    if len(opts) == 1:
        keepdim = True
    else:
        keepdim = opts[1]

    dst = resize(img, output_shape=(size[0] // opts[0], size[1] // opts[1], size[2]), order=order)

    if keepdim:
        dst = resize(dst, output_shape=(size[0], size[1], size[2]), order=order)

    return dst