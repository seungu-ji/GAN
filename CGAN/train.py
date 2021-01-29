import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torchvision import datasets

from CGAN import *
from util import *
from data_loader import *

from torchvision import transforms

import argparse

## variable setting
parser = argparse.ArgumentParser(description="CGAN parameter",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--mode", default="train", choices=["train", "test"], type=str, dest="mode")
parser.add_argument("--train_continue", default="on", choices=["on", "off"], type=str, dest="train_continue")

parser.add_argument("--lr", default=0.0002, type=float, dest="lr")
parser.add_argument("--batch_size", default=128, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=20, type=int, dest="num_epoch")

parser.add_argument("--num_class", default=10, type=int, dest="num_class")

parser.add_argument("--data_dir", default="./datasets/img_align_celeba", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default="./log", type=str, dest="log_dir")
parser.add_argument("--result_dir", default="./result", type=str, dest="result_dir")

parser.add_argument("--task", default="CGAN", choices=['CGAN'], type=str, dest="task")
parser.add_argument('--opts', nargs='+', default=['bilinear', 4.0, 0], dest='opts')

parser.add_argument("--ny", default=32, type=int, dest="ny")
parser.add_argument("--nx", default=32, type=int, dest="nx")
parser.add_argument("--nch", default=1, type=int, dest="nch")
parser.add_argument("--nker", default=128, type=int, dest="nker")

parser.add_argument("--network", default="CGAN", choices=["CGAN"], type=str, dest="network")

args = parser.parse_args()

mode = args.mode
train_continue = args.train_continue

lr = args.lr
batch_size = args.batch_size
num_epoch = args.num_epoch

num_class = args.num_class

data_dir = args.data_dir
ckpt_dir = args.ckpt_dir
log_dir = args.log_dir
result_dir = args.result_dir

task = args.task
opts = [args.opts[0], np.asarray(args.opts[1:]).astype(np.float)]

ny = args.ny
nx = args.nx
nch = args.nch
nker = args.nker

network = args.network

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# directory setting
result_dir_train = os.path.join(result_dir, 'train')
result_dir_test = os.path.join(result_dir, 'test')

if not os.path.exists(result_dir_train):
    os.makedirs(os.path.join(result_dir_train, 'png'))
if not os.path.exists(result_dir_test):
    os.makedirs(os.path.join(result_dir_test, 'png'))
    os.makedirs(os.path.join(result_dir_test, 'numpy'))

## Train
if mode == 'train':
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "./data/mnist",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(nx), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    num_batch_train = len(dataloader)
else:
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "./data/mnist",
            train=False,
            download=False,
            transform=transforms.Compose(
                [transforms.Resize(nx), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    num_batch_test = len(dataloader)

## Network setting
if network == 'CGAN':
    netG = Generator(latent_dim=100, num_class=10, nker=nker).to(device)
    netD = Discriminator(latent_dim=100, num_class=10, nker=nker).to(device)

    # 네트워크의 모든 weights 초기화 (Normal(mean=0, standard deviation=0.02))
    #init_weights(netG, init_type='normal', init_gain=0.02)
    #init_weights(netD, init_type='normal', init_gain=0.02)


## Loss function
fn_loss = nn.MSELoss().to(device)

## Optimizer
# optimizer의 momentum term B(1)값을 0.9(default) => 0.5
optimG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
optimD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

## variables setting
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean
fn_class = lambda x: 1.0 * (x > 0.5)

cmap = None

## SummaryWritter for Tensorboard
writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
writer_test = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

## TRAIN
st_epoch = 0

# TRAIN MODE
if mode == 'train':
    if train_continue == 'on':
        netG, netD, optimG, optimD, st_epoch = load(ckpt_dir=ckpt_dir, netG=netG, netD=netD, optimG=optimG, optimD=optimD)
    
    for epoch in range(st_epoch+1, num_epoch+1):
        netG.train()
        netD.train()

        loss_G_train = []
        loss_D_real_train = []
        loss_D_fake_train = []

        for batch, (imgs, labels) in enumerate(dataloader):

            batch_size = imgs.shape[0]
            # True
            valid = Variable(torch.FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False).to(device)
            # False
            fake = Variable(torch.FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False).to(device)

            # Configure input
            real_imgs = Variable(imgs.type(torch.FloatTensor)).to(device)
            labels = Variable(labels.type(torch.LongTensor)).to(device)
        
            ## Train Generator
            z = Variable(torch.FloatTensor(np.random.normal(0, 1, (batch_size, 100)))).to(device)
            g_label = Variable(torch.LongTensor(np.random.randint(0, num_class, batch_size))).to(device)

            optimG.zero_grad()

            # ouput = generated image
            output = netG(z, g_label)

            # Loss measures generator's ability to fool the discriminator
            validyity = netD(output, g_label)
            loss_G = fn_loss(validyity, valid)

            loss_G.backward()
            optimG.step()

            ## Train Discriminator
            # set_requires_grad(netD, True)
            optimD.zero_grad()
            
            pred_real = netD(real_imgs, labels) # True = torch.ones()
            pred_fake = netD(output.detach(), g_label) # False = torch.zeros()

            loss_D_real = fn_loss(pred_real, valid)
            loss_D_fake = fn_loss(pred_fake, fake)
            loss_D = (loss_D_real + loss_D_fake) / 2
            
            loss_D.backward()

            optimD.step()

            # loss function
            loss_G_train += [loss_G.item()]
            loss_D_real_train += [loss_D_real.item()]
            loss_D_fake_train += [loss_D_fake.item()]

            print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | GEN %.4f | DISC REAL: %.4f | DISC FAKE: %.4f"
                % (epoch, num_epoch, batch, num_batch_train, np.mean(loss_G_train), np.mean(loss_D_real_train), np.mean(loss_D_fake_train)))

            if batch % 20 == 0:
                # Tensorboard save
                # output은 tanh 때문에 -1 ~ 1까지의 값이 나오기 때문에 denorm 사용
                output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5)).squeeze()
                output = np.clip(output, a_min=0, a_max=1)

                id = num_batch_train * (epoch - 1) + batch

                plt.imsave(os.path.join(result_dir_train, 'png', '%04d_output.png' % id), output[0].squeeze(), cmap=cmap)

                # writer_train.add_image('output', output, id, dataformats='NHWC')

        writer_train.add_scalar('loss_G', np.mean(loss_G_train), epoch)
        writer_train.add_scalar('loss_D_real', np.mean(loss_D_real_train), epoch)
        writer_train.add_scalar('loss_D_fake', np.mean(loss_D_fake_train), epoch)

        if epoch % 2 == 0:
            save(ckpt_dir=ckpt_dir, netG=netG, netD=netD, optimG=optimG, optimD=optimD, epoch=epoch)

    writer_train.close()
# TEST MODE
else:
    netG, netD, optimG, optimD, st_epoch = load(ckpt_dir, netG=netG, netD=netD, optimG=optimG, optimD=optimD)

    with torch.no_grad():
        netG.eval()

        z = Variable(torch.FloatTensor(np.random.normal(0, 1, (batch_size, 100)))).to(device)
        g_label = Variable(torch.LongTensor(np.random.randint(0, num_class, batch_size))).to(device)

        output = netG(z, g_label)

        output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5))

        for j in range(output.shape[0]):
            id = j

            output_ = output[j]
            np.save(os.path.join(result_dir_test, 'numpy', '%04d_output.npy' % id), output_)

            #output_ = np.clip(output_, a_min=0, a_max=1)
            #plt.imsave(os.path.join(result_dir_test, 'png', '%04d_output.png' % id), output_, cmap=cmap)