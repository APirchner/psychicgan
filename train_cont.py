import argparse
import os
import json
import numpy as np

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torchvision

from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

from utils.data_utils import FramesData
from model.encoder import Encoder
from model.generator import Generator
from model.discriminator import Discrimator

def calc_gradient_penalty(netD, real_data, fake_data, l, device):
    batch_size = real_data.shape[0]
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.to(device)

    interpolates = alpha[:, None, :, None, None] * real_data + ((1 - alpha[:, None, :, None, None]) * fake_data)

    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates, _, _ = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * l
    return gradient_penalty


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data set args
    parser.add_argument('-d', '--dir', type=str, required=True, help='The data directory')
    parser.add_argument('-l', '--logdir', type=str, required=True, help='The log directory')
    # optimizer args
    parser.add_argument('-e', '--iterations', type=int, default=10, help='The number of iterations')
    # CUDA
    parser.add_argument('-c', '--disable-cuda', action='store_true', help='Disable CUDA')

    args = parser.parse_args()

    # check if logdir exists
    if not os.path.exists(args.logdir):
        raise ValueError('Logdir you\'re trying to load models from does not exist!')
    else:
        new_logdir = os.path.join(args.logdir, 'continued_training')
        os.makedirs(new_logdir)

    with open(os.path.join(args.logdir, 'args.txt'), 'r') as f:
        train_args = json.load(f)

    train_ins = train_args['ins']
    train_outs = train_args['outs']
    train_lat_dim = train_args['latent_dim']
    train_config = train_args['config']
    train_wgan = train_args['wasserstein']
    train_batch_size = train_args['batch_size']
    train_lr_discriminator = train_args['lr_discriminator']
    train_lr_generator = train_args['lr_generator']
    train_lr_encoder = train_args['lr_encoder']
    train_workers = train_args['workers']
    train_iterations = train_args['iterations']

    device = None
    if not args.disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(device)

    all_data = FramesData(train_ins, train_outs, args.dir)

    train_indices = torch.load(os.path.join(args.logdir, 'data_idx.pth'))
    train_idx = train_indices['train_idx']
    val_idx = train_indices['val_idx']
    test_idx = train_indices['test_idx']

    train_data = data.Subset(all_data, train_idx)
    val_data = data.Subset(all_data, val_idx)
    test_data = data.Subset(all_data, test_idx)

    train_loader = data.DataLoader(train_data, batch_size=train_batch_size, shuffle=True, num_workers=train_workers)
    val_loader = data.DataLoader(val_data, batch_size=32, shuffle=False, num_workers=train_workers)

    # training objectives
    disc_loss_fun = nn.BCEWithLogitsLoss()  # true-fake loss for discriminator
    gen_loss_fun = nn.MSELoss()  # feature matching loss for generator/encoder

    if not train_wgan:
        if train_config == 1:
            # basic
            encoder = Encoder(frame_dim=64, init_temp=train_ins, hidden_dim=train_lat_dim, filters=[16, 32, 64, 128],
                              attention_at=None, norm=nn.utils.weight_norm, residual=True)
            generator = Generator(frame_dim=64, temporal_target=train_outs, hidden_dim=train_lat_dim,
                                  filters=[256, 128, 64, 32], attention_at=32, norm=nn.utils.weight_norm)
            discriminator = Discrimator(frame_dim=64, init_temp=train_outs, feature_dim=128, filters=[16, 32, 64, 128],
                                        attention_at=32, norm=nn.utils.weight_norm, residual=True)
        elif train_config == 2:
            # basic - residual connections
            encoder = Encoder(frame_dim=64, init_temp=train_ins, hidden_dim=train_lat_dim, filters=[16, 32, 64, 128],
                              attention_at=None, norm=nn.utils.weight_norm, residual=False)
            generator = Generator(frame_dim=64, temporal_target=train_outs, hidden_dim=train_lat_dim,
                                  filters=[256, 128, 64, 32], attention_at=32, norm=nn.utils.weight_norm)
            discriminator = Discrimator(frame_dim=64, init_temp=train_outs, feature_dim=128, filters=[16, 32, 64, 128],
                                        attention_at=32, norm=nn.utils.weight_norm, residual=False)
        elif train_config == 3:
            # basic + spectral norm
            encoder = Encoder(frame_dim=64, init_temp=train_ins, hidden_dim=train_lat_dim, filters=[16, 32, 64, 128],
                              attention_at=None, norm=nn.utils.spectral_norm, residual=True)
            generator = Generator(frame_dim=64, temporal_target=train_outs, hidden_dim=train_lat_dim,
                                  filters=[256, 128, 64, 32], attention_at=32, norm=nn.utils.spectral_norm)
            discriminator = Discrimator(frame_dim=64, init_temp=train_outs, feature_dim=128, filters=[16, 32, 64, 128],
                                        attention_at=32, norm=nn.utils.spectral_norm, residual=True)
        elif train_config == 4:
            # basic + more filters enc/disc + spectral norm
            encoder = Encoder(frame_dim=64, init_temp=train_ins, hidden_dim=train_lat_dim, filters=[32, 64, 128, 256],
                              attention_at=None, norm=nn.utils.spectral_norm, residual=True)
            generator = Generator(frame_dim=64, temporal_target=train_outs, hidden_dim=train_lat_dim,
                                  filters=[256, 128, 64, 32], attention_at=32, norm=nn.utils.spectral_norm)
            discriminator = Discrimator(frame_dim=64, init_temp=train_outs, feature_dim=128,
                                        filters=[32, 64, 128, 256],
                                        attention_at=32, norm=nn.utils.spectral_norm, residual=True)
        elif train_config == 5:
            # basic + different capacities + spectral norm
            encoder = Encoder(frame_dim=64, init_temp=train_ins, hidden_dim=train_lat_dim, filters=[32, 64, 128, 256],
                              attention_at=None, norm=nn.utils.spectral_norm, residual=True)
            generator = Generator(frame_dim=64, temporal_target=train_outs, hidden_dim=train_lat_dim,
                                  filters=[256, 128, 64, 32], attention_at=32, norm=nn.utils.spectral_norm)
            discriminator = Discrimator(frame_dim=64, init_temp=train_outs, feature_dim=128,
                                        filters=[16, 32, 64, 128],
                                        attention_at=32, norm=nn.utils.spectral_norm, residual=True)
    else:
        if train_config == 1:
            # basic
            encoder = Encoder(frame_dim=64, init_temp=train_ins, hidden_dim=train_lat_dim, filters=[16, 32, 64, 128],
                              attention_at=16, norm=None, batchnorm=True, dropout=0.0, residual=True)
            generator = Generator(frame_dim=64, temporal_target=train_outs, hidden_dim=train_lat_dim,
                                  filters=[256, 128, 64, 32], attention_at=None, norm=None, batchnorm=True)
            discriminator = Discrimator(frame_dim=64, init_temp=1 + train_outs, target_temp=1, feature_dim=1,
                                        filters=[16, 32, 64, 128], attention_at=16, norm=None, batchnorm=False,
                                        dropout=0.0, residual=True)
        elif train_config == 2:
            # basic - attention
            encoder = Encoder(frame_dim=64, init_temp=train_ins, hidden_dim=train_lat_dim, filters=[16, 32, 64, 128],
                              attention_at=None, norm=None, batchnorm=True, dropout=0.0, residual=True)
            generator = Generator(frame_dim=64, temporal_target=train_outs, hidden_dim=train_lat_dim,
                                  filters=[256, 128, 64, 32], attention_at=None, norm=None, batchnorm=True)
            discriminator = Discrimator(frame_dim=64, init_temp=1 + train_outs, target_temp=1, feature_dim=1,
                                        filters=[16, 32, 64, 128], attention_at=None, norm=None, batchnorm=False,
                                        dropout=0.0, residual=True)
        elif train_config == 3:
            # basic + more filters enc/disc
            encoder = Encoder(frame_dim=64, init_temp=train_ins, hidden_dim=train_lat_dim, filters=[32, 64, 128, 256],
                              attention_at=16, norm=None, batchnorm=True, dropout=0.0, residual=True)
            generator = Generator(frame_dim=64, temporal_target=train_outs, hidden_dim=train_lat_dim,
                                  filters=[256, 128, 64, 32], attention_at=None, norm=None, batchnorm=True)
            discriminator = Discrimator(frame_dim=64, init_temp=1 + train_outs, target_temp=1, feature_dim=1,
                                        filters=[32, 64, 128, 256], attention_at=16, norm=None, batchnorm=False,
                                        dropout=0.0, residual=True)
        elif train_config == 4:
            # basic + more filters enc/disc + spectral norm
            encoder = Encoder(frame_dim=64, init_temp=train_ins, hidden_dim=train_lat_dim, filters=[32, 64, 128, 256],
                              attention_at=None, norm=nn.utils.spectral_norm, residual=True)
            generator = Generator(frame_dim=64, temporal_target=train_outs, hidden_dim=train_lat_dim,
                                  filters=[256, 128, 64, 32], attention_at=32, norm=nn.utils.spectral_norm)
            discriminator = Discrimator(frame_dim=64, init_temp=train_outs, feature_dim=1,
                                        filters=[32, 64, 128, 256],
                                        attention_at=32, norm=nn.utils.spectral_norm, batchnorm=False, residual=True)
        elif train_config == 5:
            # basic + different capacities + spectral norm
            encoder = Encoder(frame_dim=64, init_temp=train_ins, hidden_dim=train_lat_dim, filters=[32, 64, 128, 256],
                              attention_at=None, norm=nn.utils.spectral_norm, residual=True)
            generator = Generator(frame_dim=64, temporal_target=train_outs, hidden_dim=train_lat_dim,
                                  filters=[256, 128, 64, 32], attention_at=32, norm=nn.utils.spectral_norm)
            discriminator = Discrimator(frame_dim=64, init_temp=train_outs, feature_dim=1,
                                        filters=[16, 32, 64, 128],
                                        attention_at=32, norm=nn.utils.spectral_norm, batchnorm=False, residual=True)

    encoder.load_state_dict(torch.load(os.path.join(args.logdir, 'encoder.pth'), map_location=device))
    generator.load_state_dict(torch.load(os.path.join(args.logdir, 'generator.pth'), map_location=device))
    discriminator.load_state_dict(torch.load(os.path.join(args.logdir, 'discriminator.pth'), map_location=device))

    encoder = encoder.to(device)
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    # print model summaries
    summary(encoder, input_size=(3, train_ins, 64, 64))
    summary(generator, input_size=(train_lat_dim,))
    summary(discriminator, input_size=(3, 1 + train_outs, 64, 64))

    # tensorboard log writer
    tb_writer = SummaryWriter(log_dir=new_logdir)

    if train_wgan:
        # WASSERSTEIN GAN

        # in WGAN, moment-based optimizers don't appear to work as well -> see WGAN paper
        # encoder_optim = optim.RMSprop(encoder.parameters(), lr=args.lr_encoder)
        # generator_optim = optim.RMSprop(generator.parameters(), lr=args.lr_generator)
        # discriminator_optim = optim.RMSprop(discriminator.parameters(), lr=args.lr_discriminator)
        # in improved WGAN they use Adam
        encoder_optim = optim.Adam(encoder.parameters(), lr=train_lr_encoder, betas=(0., 0.9))
        generator_optim = optim.Adam(generator.parameters(), lr=train_lr_generator, betas=(0., 0.9))
        discriminator_optim = optim.Adam(discriminator.parameters(), lr=train_lr_discriminator, betas=(0., 0.9))

        # encoder_sched = optim.lr_scheduler.ExponentialLR(encoder_optim, gamma=0.99)
        # generator_sched = optim.lr_scheduler.ExponentialLR(generator_optim, gamma=0.99)
        # discriminator_sched = optim.lr_scheduler.ExponentialLR(discriminator_optim, gamma=0.99)

        one = torch.FloatTensor([1]).to(device)
        m_one = -1 * one

        i = 0
        train_iter = iter(train_loader)
        for global_step in range(args.iterations):

            for j in range(5):
                # do 5 discriminator steps for each encoder/generator step -> see WGAN paper
                # get the inputs and move them to device
                data = next(train_iter)
                i += 1
                if i == len(train_loader):
                    train_iter = iter(train_loader)
                    i = 0
                in_frames, out_frames = data
                in_frames = in_frames.to(device)
                out_frames = out_frames.to(device)

                # zero the parameter gradients
                encoder.zero_grad()
                generator.zero_grad()
                discriminator.zero_grad()

                # DISCRIMINATOR TRAINING
                with torch.no_grad():
                    hidden, _ = encoder(in_frames)
                    generated, _ = generator(hidden)
                # add one last past frame to the discriminator input
                out_frames_disc = torch.cat([in_frames[:, :, -1, :, :].unsqueeze(2), out_frames], dim=2)
                generated_disc = torch.cat([in_frames[:, :, -1, :, :].unsqueeze(2), generated], dim=2)

                out_real, _, _ = discriminator(out_frames_disc)
                out_gen, _, _ = discriminator(generated_disc)

                err_real = out_real.mean(0).view(1)
                err_gen = out_gen.mean(0).view(1)

                loss_D = err_gen - err_real + calc_gradient_penalty(discriminator, out_frames_disc, generated_disc, 10,
                                                                    device)
                loss_D.backward()
                # err_real.backward(m_one)
                # err_gen.backward(one)
                # err_D = err_real + err_gen
                discriminator_optim.step()

            # GENERATOR/ENCODER TRAINING
            data = next(train_iter)
            i += 1
            if i == len(train_loader):
                train_iter = iter(train_loader)
                i = 0
            in_frames, out_frames = data
            in_frames = in_frames.to(device)
            out_frames = out_frames.to(device)

            encoder.zero_grad()
            generator.zero_grad()
            discriminator.zero_grad()

            hidden, encoder_attn = encoder(in_frames)
            generated, generator_attn = generator(hidden)

            generated_disc = torch.cat([in_frames[:, :, -1, :, :].unsqueeze(2), generated], dim=2)

            out_gen, _, _ = discriminator(generated_disc)
            err_G = out_gen.mean(0).view(1)

            loss_G = -err_G
            loss_G.backward()
            # err_G.backward(one)
            generator_optim.step()
            encoder_optim.step()

            # print statistics
            if global_step % 10 == 9:
                print('[Step {0}] Loss: (D) {1} - (G) {2}'.format(
                    global_step, round(loss_D.item(), 4), round(loss_G.item(), 4)))
                tb_writer.add_scalar('D_loss', loss_D.item(), global_step=global_step+train_iterations)
                tb_writer.add_scalar('G_loss', loss_G.item(), global_step=global_step+train_iterations)

            if global_step % 100 == 99:
                # log generated and real images
                gen_imgs = torchvision.utils.make_grid(((generated[:64, :, :, :, :]+1)/2).squeeze())
                real_imgs = torchvision.utils.make_grid(((out_frames[:64, :, :, :, :]+1)/2).squeeze())
                tb_writer.add_image('G_imgs', gen_imgs, global_step=global_step+train_iterations)
                tb_writer.add_image('R_imgs', real_imgs, global_step=global_step+train_iterations)

            # do the validation
            if global_step % 100 == 99:
                generator.train(False)
                encoder.train(False)
                discriminator.train(False)
                val_loss = 0.0
                for in_frames, out_frames in val_loader:
                    with torch.no_grad():
                        encoder.zero_grad()
                        generator.zero_grad()
                        discriminator.zero_grad()
                        # get the validation inputs and outputs
                        in_frames = in_frames.to(device)
                        out_frames = out_frames.to(device)

                        # forward
                        hidden, _ = encoder(in_frames)
                        generated, _ = generator(hidden)

                        out_frames_disc = torch.cat([in_frames[:, :, -1, :, :].unsqueeze(2), out_frames], dim=2)
                        generated_disc = torch.cat([in_frames[:, :, -1, :, :].unsqueeze(2), generated], dim=2)

                        out_real, _, _ = discriminator(out_frames_disc)
                        out_gen, _, _ = discriminator(generated_disc)

                        err_real = out_real.mean(0).view(1)
                        err_gen = out_gen.mean(0).view(1)

                        loss_D = err_gen - err_real
                        # calc_gradient_penalty(discriminator, out_frames_disc, generated_disc, 10, device)
                        val_loss += loss_D/len(val_data)
                print('[Step {0}] Val-loss: (D) {1}'.format(
                    global_step, round(val_loss.item(), 4)))
                tb_writer.add_scalar('val_loss', val_loss.item(), global_step=global_step+train_iterations)
                gen_imgs_val = torchvision.utils.make_grid(((generated + 1) / 2).squeeze())
                real_imgs_val = torchvision.utils.make_grid(((out_frames + 1) / 2).squeeze())
                tb_writer.add_image('G_imgs_val', gen_imgs_val, global_step=global_step+train_iterations)
                tb_writer.add_image('R_imgs_val', real_imgs_val, global_step=global_step+train_iterations)
                generator.train(True)
                encoder.train(True)
                discriminator.train(True)

    # write model to disk
    torch.save(encoder.state_dict(), os.path.join(new_logdir, 'encoder.pth'))
    torch.save(generator.state_dict(), os.path.join(new_logdir, 'generator.pth'))
    torch.save(discriminator.state_dict(), os.path.join(new_logdir, 'discriminator.pth'))
    torch.save({'train_idx': train_idx, 'val_idx': val_idx, 'test_idx': test_idx},
               os.path.join(new_logdir, 'data_idx.pth'))

