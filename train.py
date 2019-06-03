import argparse

import matplotlib.pyplot as plt
import torchvision.utils as vutils

import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim

from torchsummary import summary

from utils.data_utils import *
from model.encoder import Encoder
from model.generator import Generator
from model.discriminator import Discrimator

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # data set args
    parser.add_argument('-d', '--dir', type=str, required=True, help='The data directory')
    parser.add_argument('-p', '--sample_dir', type=str, required=True, help='The generated sample output dir')
    parser.add_argument('-i', '--ins', type=int, required=True, help='The number of conditioning frames')
    parser.add_argument('-o', '--outs', type=int, required=True, help='The number of generated frames')
    # optimizer args
    parser.add_argument('-e', '--epochs', type=int, default=20, help='The number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='The batch size')
    parser.add_argument('-l', '--lr_generator', type=float, default=1e-4, help='The generator learning rate')
    parser.add_argument('-m', '--lr_encoder', type=float, default=1e-4, help='The encoder learning rate')
    parser.add_argument('-n', '--lr_discriminator', type=float, default=1e-4, help='The discriminator learning rate')
    # CUDA
    parser.add_argument('-c', '--disable-cuda', action='store_true', help='Disable CUDA')

    args = parser.parse_args()
    print(args)

    device = None
    if not args.disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(device)

    all_data = KITTIData(args.ins, args.outs, 0, args.dir)
    [train_data,val_data] = data.random_split(all_data,[1100,243])
    train_loader = data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=16)
    val_loader = data.DataLoader(val_data, batch_size = 1, shuffle = False, num_workers = 1)

    loss_fun = nn.MSELoss()

    encoder = Encoder(frame_dim=64, init_temp=2, hidden_dim=128, out_filters=256,
                      attention_at=8, norm=nn.utils.weight_norm, residual=True)
    encoder = encoder.to(device)
    encoder_optim = optim.Adam(encoder.parameters(), betas=(0.9, 0.999))

    generator = Generator(frame_dim=64, temporal_target=1, hidden_dim=128,
                          init_filters=256, attention_at=32, norm=nn.utils.weight_norm)
    generator = generator.to(device)
    generator_optim = optim.Adam(generator.parameters(), betas=(0.5, 0.999))

    discriminator = Discrimator(frame_dim=64, init_temp=1, feature_dim=128,
                                out_filters=256, attention_at=8, norm=nn.utils.weight_norm)
    discriminator = discriminator.to(device)
    discriminator_optim = optim.Adam(discriminator.parameters(), betas=(0.9, 0.999))

    summary(encoder, input_size=(3, args.ins, 64, 64))
    summary(generator, input_size=(128,))
    summary(discriminator, input_size=(3, args.ins, 64, 64))

    for epoch in range(args.epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs and move them to device
            in_frames, out_frames = data
            in_frames = in_frames.to(device)
            out_frames = out_frames.to(device)

            # zero the parameter gradients
            encoder_optim.zero_grad()
            generator_optim.zero_grad()
            discriminator.zero_grad()

            # DISCRIMINATOR TRAINING
            # forward + backward + optimize
            with torch.no_grad():
                hidden, _ = encoder(in_frames)
                generated, _ = generator(hidden)
            features_real, disc_attn_real = discriminator(out_frames)
            features_gen, disc_attn_gen = discriminator(generated)
            loss = loss_fun(features_real, features_gen)

            loss.backward()
            discriminator_optim.step()

            # GENERATOR/ENCODER TRAINING
            hidden, encoder_attn = encoder(in_frames)
            generated, generator_attn = generator(hidden)

            features_real, disc_attn_real = discriminator(out_frames)
            features_gen, disc_attn_gen = discriminator(generated)

            loss = loss_fun(features_real, features_gen)

            encoder_optim.step()
            generator_optim.step()

            # print statistics
            running_loss += loss.item() / 10
            if i % 10 == 9:
                print('[Epoch {0} - Step {1}] Loss: {2}'.format(epoch, i, running_loss))
                running_loss = 0
                #plt.imsave('/home/andreas/Documents/msc_info/sem_2/adl4cv/kitti/test/epoch{0}step{1}.jpg'.format(epoch, i),
                #           np.transpose(generated[1].squeeze().detach().cpu()))

        val_loss = 0.0
        for inval, outval in val_loader:
            # get the validation inputs and outputs
            inval, outval = inval.to(device), outval.to(device)

            # forward
            hidval, encval_attn = encoder(inval)
            genval, genval_attn = generator(hidval)
            val_loss += loss_fun(genval, outval).item() / len(val_loader)

        print('[Epoch {0}] Val-Loss: {1}'.format(epoch, val_loss))
