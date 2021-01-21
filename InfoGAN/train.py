import os
import torch
import torchvision
import numpy as np

from InfoGAN import *


# MNIST
dataset = ['']

if dataset == 'MNIST':
    mode = 'train'
    train_continue = 'on'

    img_size = 28 
    z_dim = 62 # noise variable
    num_epoch = 30 # or 50
    batch_size = 128
    lr_D = 0.0002 # Disriminator learning rate
    lr_G = 0.001 # Generator learning rate

    cc_dim = 2 # Continous latent codes
    dc_dim = 1 # ten_dimensional categorical code = 10 x dimensional code
    continuous_weight = 0.5 # 0.5 ~ 1

    dataset = 'MNIST'
    data_dir = './dataset/img_align_celeba'
    ckpt_dir = './checkpoint'
    log_dir = './log'
    result_dir = './result'

    network = 'InfoGAN'



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# directory setting
result_dir_train = os.path.join(result_dir, 'train')
result_dir_test = os.path.join(result_dir, 'test')

if not os.path.exists(result_dir_train):
    os.makedirs(os.path.join(result_dir_train, 'png'))
if not os.path.exists(result_dir_test):
    os.makedirs(os.path.join(result_dir_test, 'png'))
    os.makedirs(os.path.join(result_dir_test, 'numpy'))


## TRAIN
"""
if mode == 'train':
    transform_train = transforms.Compose([Resize(shape=(ny, nx, nch)), Normalization(mean=0.5, std=0.5)])

    dataset_train = Dataset(data_dir=data_dir, transform=transform_train, task=task, opts=opts)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)

    num_data_train = len(dataset_train)
    num_batch_train = np.ceil(num_data_train / batch_size)
else:
    transform_test = transforms.Compose([RandomCrop(shape=(ny, nx))])

    dataset_test = Dataset(data_dir=data_dir, transform=transform_test, task=task, opts=opts)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=4)

    num_data_test = len(dataset_test)
    num_batch_test = np.ceil(num_data_test / batch_size)
"""

## Network setting
if network == 'DCGAN':
    netG = Generator(dataset, z_dim, cc_dim, dc_dim).to(device)
    netD = Discriminator(dataset, cc_dim, dc_dim).to(device)
    """
    # 네트워크의 모든 weights 초기화 (Normal(mean=0, standard deviation=0.02))
    init_weights(netG, init_type='normal', init_gain=0.02)
    init_weights(netD, init_type='normal', init_gain=0.02)
    """

if mode == 'train':
    if train_continue == 'on':
        pass

    for 