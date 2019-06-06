import argparse

import torch.nn as nn
import torch.optim as optim
import torchvision

from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

from utils.data_utils import *
from model.encoder import Encoder
from model.generator import Generator
from model.discriminator import Discrimator


def weight_init(net):
    classname = net.__class__.__name__
    if classname.find('Conv3d') != -1:
        nn.init.normal_(net.weight.data, 0.0, 0.02)
        if classname.find('Conv1d') != -1:
            nn.init.normal_(net.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(net.weight.data, 1.0, 0.02)
        nn.init.constant_(net.bias.data, 0)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # data set args
    parser.add_argument('-d', '--dir', type=str, required=True, help='The data directory')
    parser.add_argument('-i', '--ins', type=int, required=True, help='The number of conditioning frames')
    parser.add_argument('-o', '--outs', type=int, required=True, help='The number of generated frames')
    parser.add_argument('-z', '--latent_dim', type=int, default=128, help='The dimension of the latent frame encoding')
    # optimizer args
    parser.add_argument('-e', '--epochs', type=int, default=20, help='The number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='The batch size')
    parser.add_argument('-l', '--lr_generator', type=float, default=1e-4, help='The generator learning rate')
    parser.add_argument('-m', '--lr_encoder', type=float, default=1e-4, help='The encoder learning rate')
    parser.add_argument('-n', '--lr_discriminator', type=float, default=4e-4, help='The discriminator learning rate')
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
    [train_data, val_data] = data.random_split(all_data, [1100, 243])
    train_loader = data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=16)
    val_loader = data.DataLoader(val_data, batch_size=1, shuffle=False, num_workers=1)

    # training objectives
    disc_loss_fun = nn.BCEWithLogitsLoss()  # true-fake loss for discriminator
    gen_loss_fun = nn.MSELoss()  # feature matching loss for generator/encoder

    # encoder setup
    encoder = Encoder(frame_dim=64, init_temp=2, hidden_dim=args.latent_dim, filters=[32, 64, 64, 128],
                      attention_at=None, norm=nn.utils.weight_norm, residual=True)
    encoder = encoder.to(device)
    encoder.apply(weight_init)
    encoder_optim = optim.Adam(encoder.parameters(), lr=args.lr_encoder, betas=(0.0, 0.9))

    # generator setup
    generator = Generator(frame_dim=64, temporal_target=1, hidden_dim=args.latent_dim,
                          filters=[256, 128, 64, 32], attention_at=32, norm=nn.utils.weight_norm)
    generator = generator.to(device)
    generator.apply(weight_init)
    generator_optim = optim.Adam(generator.parameters(), lr=args.lr_generator, betas=(0.0, 0.9))

    # discriminator setup
    discriminator = Discrimator(frame_dim=64, init_temp=1, feature_dim=128, filters=[32, 64, 64, 128],
                                attention_at=None, norm=nn.utils.weight_norm, residual=True)
    discriminator = discriminator.to(device)
    discriminator.apply(weight_init)
    discriminator_optim = optim.Adam(discriminator.parameters(), lr=args.lr_discriminator, betas=(0.0, 0.9))

    # print model summaries
    summary(encoder, input_size=(3, args.ins, 64, 64))
    summary(generator, input_size=(128,))
    summary(discriminator, input_size=(3, args.ins, 64, 64))

    # tensorboard log writer
    tb_writer = SummaryWriter(log_dir='/home/andreas/Documents/msc_info/sem_2/adl4cv/runs')

    # training loop
    global_step = 0
    for epoch in range(args.epochs):
        disc_running_loss = 0.0
        gen_running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs and move them to device
            in_frames, out_frames = data
            in_frames = in_frames.to(device)
            out_frames = out_frames.to(device)

            # target for discriminator training
            batch_size = in_frames.shape[0]
            y_real = torch.ones((batch_size, 1), dtype=torch.float32).to(device)
            y_gen = torch.zeros((batch_size, 1), dtype=torch.float32).to(device)

            # zero the parameter gradients
            encoder_optim.zero_grad()
            generator_optim.zero_grad()
            discriminator.zero_grad()

            # DISCRIMINATOR TRAINING
            hidden, _ = encoder(in_frames)
            generated, _ = generator(hidden)
            _, logits_real, _ = discriminator(out_frames)
            _, logits_gen, _ = discriminator(generated.detach())
            # loss on real batch
            disc_loss_real = disc_loss_fun(logits_real, y_real)
            # loss on fake batch
            disc_loss_gen = disc_loss_fun(logits_gen, y_gen)

            disc_loss = (disc_loss_real + disc_loss_gen) / 2
            disc_loss.backward()

            discriminator_optim.step()

            # GENERATOR/ENCODER TRAINING
            hidden, encoder_attn = encoder(in_frames)
            generated, generator_attn = generator(hidden)

            features_real, logits_real, disc_attn_real = discriminator(out_frames)
            features_gen, logits_gen, disc_attn_gen = discriminator(generated)

            gen_loss = gen_loss_fun(torch.mean(features_real, dim=0), torch.mean(features_gen, dim=0))
            gen_loss.backward()

            encoder_optim.step()
            generator_optim.step()

            # print statistics
            disc_running_loss += disc_loss.item()
            gen_running_loss += gen_loss.item()
            if i % 10 == 9:
                print('[Epoch {0} - Step {1}] Loss: (D) {2} - (G) {3}'.format(
                    epoch, i, disc_running_loss / 10, gen_running_loss / 10))
                tb_writer.add_scalar('D_loss_real', disc_running_loss / 10, global_step=global_step)
                tb_writer.add_scalar('D_loss_gen', disc_running_loss / 10, global_step=global_step)
                tb_writer.add_scalar('G_loss', gen_running_loss / 10, global_step=global_step)
                # log generated images
                gen_imgs = torchvision.utils.make_grid(generated.squeeze())
                tb_writer.add_image('G_imgs', gen_imgs, global_step=global_step)

            global_step += 1

        # val_loss = 0.0
        # for inval, outval in val_loader:
        #     # get the validation inputs and outputs
        #     inval, outval = inval.to(device), outval.to(device)
        #
        #     # forward
        #     hidval, encval_attn = encoder(inval)
        #     genval, genval_attn = generator(hidval)
        #     val_loss += loss_fun(genval, outval).item() / len(val_loader)
        #
        # print('[Epoch {0}] Val-Loss: {1}'.format(epoch, val_loss))
