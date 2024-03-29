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
from model.encoder import Encoder, EncoderMoreConvs
from model.generator import Generator, GeneratorMoreConvs
from model.discriminator import Discrimator, DiscrimatorMoreConvs


def weight_init(net):
    """
    Initializes a given neural net with the Xavier normal dist.
    :param net:  the network to initialize
    :return: None
    """
    classname = net.__class__.__name__
    if classname.find('Conv3d') != -1:
        nn.init.xavier_normal_(net.weight.data)
    elif classname.find('Conv1d') != -1:
        nn.init.xavier_normal_(net.weight.data)


def calc_gradient_penalty(netD, real_data, fake_data, l, device):
    """
    Calculates the gradient penalty defined in "Improved Training of Wasserstein GANs".
    :param netD: the discriminator
    :param real_data: batch of real frames
    :param fake_data: batch of generated frames
    :param l: penalty strength lambda
    :param device: cuda or cpu
    :return: the penalty value
    """
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


#### MAIN ####

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data set args
    parser.add_argument('-d', '--dir', type=str, required=True, help='The data directory')
    parser.add_argument('-l', '--logdir', type=str, required=True, help='The log directory')
    parser.add_argument('-i', '--ins', type=int, required=True, help='The number of conditioning frames')
    parser.add_argument('-o', '--outs', type=int, required=True, help='The number of generated frames')
    parser.add_argument('-z', '--latent_dim', type=int, default=128, help='The dimension of the latent frame encoding')
    parser.add_argument('-w', '--wasserstein', action='store_true', help='Use Wasserstein GAN')
    parser.add_argument('-v', '--config', type=int, default=1, help='The model configuration', choices=[1, 2, 3, 4, 5])
    # optimizer args
    parser.add_argument('-e', '--iterations', type=int, default=10, help='The number of iterations')
    parser.add_argument('-t', '--workers', type=int, default=4, help='The number of threads for data pre-fetching')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='The batch size')
    parser.add_argument('-g', '--lr_generator', type=float, default=1e-4, help='The generator learning rate')
    parser.add_argument('-m', '--lr_encoder', type=float, default=1e-4, help='The encoder learning rate')
    parser.add_argument('-n', '--lr_discriminator', type=float, default=1e-4, help='The discriminator learning rate')
    # CUDA
    parser.add_argument('-c', '--disable-cuda', action='store_true', help='Disable CUDA')

    args = parser.parse_args()

    # set up log directory
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    with open(os.path.join(args.logdir, 'args.txt'), 'w') as f:
        json.dump(vars(args), f, indent=4, separators=(',', ':'))

    device = None
    if not args.disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(device)

    all_data = FramesData(args.ins, args.outs, args.dir)

    # train/evaluation/test split
    shuffled_idx = list(range(len(all_data)))
    np.random.shuffle(shuffled_idx)
    train_idx = shuffled_idx[:80000]
    val_idx = shuffled_idx[80000:90000]
    test_idx = shuffled_idx[90000:100000]

    train_data = data.Subset(all_data, train_idx)
    val_data = data.Subset(all_data, val_idx)

    train_loader = data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = data.DataLoader(val_data, batch_size=32, shuffle=False, num_workers=args.workers)

    # training objectives
    disc_loss_fun = nn.BCEWithLogitsLoss()  # true-fake loss for discriminator
    gen_loss_fun = nn.MSELoss()  # feature matching loss for generator/encoder

    # architecture configs for FM ang WGAN
    if not args.wasserstein:
        if args.config == 1:
            # basic
            encoder = Encoder(frame_dim=64, init_temp=args.ins, hidden_dim=args.latent_dim, filters=[16, 32, 64, 128],
                              attention_at=None, norm=nn.utils.weight_norm, residual=True)
            generator = Generator(frame_dim=64, temporal_target=args.outs, hidden_dim=args.latent_dim,
                                  filters=[256, 128, 64, 32], attention_at=32, norm=nn.utils.weight_norm)
            discriminator = Discrimator(frame_dim=64, init_temp=args.outs, feature_dim=128, filters=[16, 32, 64, 128],
                                        attention_at=32, norm=nn.utils.weight_norm, residual=True)
        elif args.config == 2:
            # basic - residual connections
            encoder = Encoder(frame_dim=64, init_temp=args.ins, hidden_dim=args.latent_dim, filters=[16, 32, 64, 128],
                              attention_at=None, norm=nn.utils.weight_norm, residual=False)
            generator = Generator(frame_dim=64, temporal_target=args.outs, hidden_dim=args.latent_dim,
                                  filters=[256, 128, 64, 32], attention_at=32, norm=nn.utils.weight_norm)
            discriminator = Discrimator(frame_dim=64, init_temp=args.outs, feature_dim=128, filters=[16, 32, 64, 128],
                                        attention_at=32, norm=nn.utils.weight_norm, residual=False)
        elif args.config == 3:
            # basic + spectral norm
            encoder = Encoder(frame_dim=64, init_temp=args.ins, hidden_dim=args.latent_dim, filters=[16, 32, 64, 128],
                              attention_at=None, norm=nn.utils.spectral_norm, residual=True)
            generator = Generator(frame_dim=64, temporal_target=args.outs, hidden_dim=args.latent_dim,
                                  filters=[256, 128, 64, 32], attention_at=32, norm=nn.utils.spectral_norm)
            discriminator = Discrimator(frame_dim=64, init_temp=args.outs, feature_dim=128, filters=[16, 32, 64, 128],
                                        attention_at=32, norm=nn.utils.spectral_norm, residual=True)
        elif args.config == 4:
            # basic + more filters enc/disc + spectral norm
            encoder = Encoder(frame_dim=64, init_temp=args.ins, hidden_dim=args.latent_dim, filters=[32, 64, 128, 256],
                              attention_at=None, norm=nn.utils.spectral_norm, residual=True)
            generator = Generator(frame_dim=64, temporal_target=args.outs, hidden_dim=args.latent_dim,
                                  filters=[256, 128, 64, 32], attention_at=32, norm=nn.utils.spectral_norm)
            discriminator = Discrimator(frame_dim=64, init_temp=args.outs, feature_dim=128,
                                        filters=[32, 64, 128, 256],
                                        attention_at=32, norm=nn.utils.spectral_norm, residual=True)
        elif args.config == 5:
            # basic + different capacities + spectral norm
            encoder = Encoder(frame_dim=64, init_temp=args.ins, hidden_dim=args.latent_dim, filters=[32, 64, 128, 256],
                              attention_at=None, norm=nn.utils.spectral_norm, residual=True)
            generator = Generator(frame_dim=64, temporal_target=args.outs, hidden_dim=args.latent_dim,
                                  filters=[256, 128, 64, 32], attention_at=32, norm=nn.utils.spectral_norm)
            discriminator = Discrimator(frame_dim=64, init_temp=args.outs, feature_dim=128,
                                        filters=[16, 32, 64, 128],
                                        attention_at=32, norm=nn.utils.spectral_norm, residual=True)
    else:
        if args.config == 1:
            # basic
            encoder = EncoderMoreConvs(frame_dim=64, init_temp=args.ins, hidden_dim=args.latent_dim,
                                       filters=[16, 32, 64, 128], attention_at=16, target_temp=1,
                                       norm=None, batchnorm=True, dropout=0.0, residual=True)
            generator = GeneratorMoreConvs(frame_dim=64, temporal_target=args.outs, hidden_dim=args.latent_dim,
                                           filters=[256, 128, 64, 32], attention_at=None, norm=None, batchnorm=True)
            discriminator = Discrimator(frame_dim=64, init_temp=1 + args.outs, target_temp=1, feature_dim=1,
                                        filters=[16, 32, 64, 128], attention_at=16, norm=None, batchnorm=False,
                                        dropout=0.0, residual=True)
        elif args.config == 2:
            # basic - attention
            encoder = Encoder(frame_dim=64, init_temp=args.ins, hidden_dim=args.latent_dim, filters=[16, 32, 64, 128],
                              attention_at=None, norm=None, batchnorm=True, dropout=0.0, residual=True)
            generator = Generator(frame_dim=64, temporal_target=args.outs, hidden_dim=args.latent_dim,
                                  filters=[256, 128, 64, 32], attention_at=None, norm=None, batchnorm=True)
            discriminator = Discrimator(frame_dim=64, init_temp=1 + args.outs, target_temp=1, feature_dim=1,
                                        filters=[16, 32, 64, 128], attention_at=None, norm=None, batchnorm=False,
                                        dropout=0.0, residual=True)
        elif args.config == 3:
            # basic + more filters enc/disc
            encoder = EncoderMoreConvs(frame_dim=64, init_temp=args.ins, hidden_dim=args.latent_dim,
                                       filters=[32, 64, 128, 256], attention_at=16, norm=None, batchnorm=True,
                                       dropout=0.0, residual=True)
            generator = GeneratorMoreConvs(frame_dim=64, temporal_target=args.outs, hidden_dim=args.latent_dim,
                                           filters=[256, 128, 64, 32], attention_at=None, norm=None, batchnorm=True)
            discriminator = DiscrimatorMoreConvs(frame_dim=64, init_temp=1 + args.outs, target_temp=1, feature_dim=1,
                                                 filters=[32, 64, 128, 256], attention_at=16, norm=None,
                                                 batchnorm=False, dropout=0.0, residual=True)
        elif args.config == 4:
            # basic + more filters enc/disc + attn in gen
            encoder = EncoderMoreConvs(frame_dim=64, init_temp=args.ins, hidden_dim=args.latent_dim,
                                       filters=[32, 64, 128, 256],
                                       attention_at=None, norm=None, batchnorm=True, dropout=0.0, residual=True)
            generator = GeneratorMoreConvs(frame_dim=64, temporal_target=args.outs, hidden_dim=args.latent_dim,
                                           filters=[256, 128, 64, 32], attention_at=None, norm=None, batchnorm=True)
            discriminator = DiscrimatorMoreConvs(frame_dim=64, init_temp=1 + args.outs, target_temp=1, feature_dim=1,
                                                 filters=[32, 64, 128, 256], attention_at=None, norm=None,
                                                 batchnorm=False,
                                                 dropout=0.0, residual=True)
        elif args.config == 5:
            # basic + different capacities + spectral norm
            encoder = EncoderMoreConvs(frame_dim=64, init_temp=args.ins, hidden_dim=args.latent_dim,
                                       filters=[32, 64, 128, 256],
                                       attention_at=None, norm=nn.utils.spectral_norm, residual=True)
            generator = GeneratorMoreConvs(frame_dim=64, temporal_target=args.outs, hidden_dim=args.latent_dim,
                                           filters=[256, 128, 64, 32], attention_at=32, norm=nn.utils.spectral_norm)
            discriminator = DiscrimatorMoreConvs(frame_dim=64, init_temp=args.outs, feature_dim=1,
                                                 filters=[16, 32, 64, 128],
                                                 attention_at=16, norm=nn.utils.spectral_norm, batchnorm=False,
                                                 residual=True)

    # network setup and weight init
    encoder = encoder.to(device)
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    encoder.apply(weight_init)
    generator.apply(weight_init)
    discriminator.apply(weight_init)

    # print model summaries
    summary(encoder, input_size=(3, args.ins, 64, 64))
    summary(generator, input_size=(args.latent_dim,))
    summary(discriminator, input_size=(3, 1 + args.outs, 64, 64))

    # tensorboard log writer
    tb_writer = SummaryWriter(log_dir=args.logdir)

    if args.wasserstein:
        # WGAN training loop
        encoder_optim = optim.Adam(encoder.parameters(), lr=args.lr_encoder, betas=(0., 0.9))
        generator_optim = optim.Adam(generator.parameters(), lr=args.lr_generator, betas=(0., 0.9))
        discriminator_optim = optim.Adam(discriminator.parameters(), lr=args.lr_discriminator, betas=(0., 0.9))

        one = torch.FloatTensor([1]).to(device)
        m_one = -1 * one

        i = 0
        train_iter = iter(train_loader)
        for global_step in range(args.iterations):
            # if global_step < 10 or global_step%500 == 0:
            #     disc_steps = 100
            # else:
            #     disc_steps = 5
            disc_steps = 5
            for j in range(disc_steps):
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
            generator_optim.step()
            encoder_optim.step()

            # print statistics
            if global_step % 10 == 9:
                print('[Step {0}] Loss: (D) {1} - (G) {2}'.format(
                    global_step, round(loss_D.item(), 4), round(loss_G.item(), 4)))
                tb_writer.add_scalar('G_loss', loss_G.item(), global_step=global_step)
                tb_writer.add_scalar('D_loss', loss_D.item(), global_step=global_step)

            if global_step % 100 == 99:
                # log generated and real images
                gen_imgs = torchvision.utils.make_grid(((generated[:64, :, :, :, :] + 1) / 2).squeeze())
                real_imgs = torchvision.utils.make_grid(((out_frames[:64, :, :, :, :] + 1) / 2).squeeze())
                tb_writer.add_image('G_imgs', gen_imgs, global_step=global_step)
                tb_writer.add_image('R_imgs', real_imgs, global_step=global_step)

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
                        val_loss += loss_D / len(val_data)
                print('[Step {0}] Val-loss: (D) {1}'.format(
                    global_step, round(val_loss.item(), 4)))
                tb_writer.add_scalar('val_loss', val_loss.item(), global_step=global_step)
                gen_imgs_val = torchvision.utils.make_grid(((generated + 1) / 2).squeeze())
                real_imgs_val = torchvision.utils.make_grid(((out_frames + 1) / 2).squeeze())
                tb_writer.add_image('G_imgs_val', gen_imgs_val, global_step=global_step)
                tb_writer.add_image('R_imgs_val', real_imgs_val, global_step=global_step)
                generator.train(True)
                encoder.train(True)
                discriminator.train(True)

    else:
        # feature matching training loop
        encoder_optim = optim.Adam(encoder.parameters(), lr=args.lr_encoder, betas=(0.0, 0.9))
        generator_optim = optim.Adam(generator.parameters(), lr=args.lr_generator, betas=(0.0, 0.9))
        discriminator_optim = optim.Adam(discriminator.parameters(), lr=args.lr_discriminator, betas=(0.0, 0.9))

        # FEATURE MATCHING
        global_step = 0
        for epoch in range(args.epochs):
            disc_running_loss = 0.0
            gen_running_loss = 0.0
            disc_running_acc_real = 0.0
            disc_running_acc_gen = 0.0
            for i, data in enumerate(train_loader, 0):
                # get the inputs and move them to device
                in_frames, out_frames = data
                in_frames = in_frames.to(device)
                out_frames = out_frames.to(device)

                # target for discriminator training
                batch_size = in_frames.shape[0]
                y_gen = torch.ones((batch_size, 1), dtype=torch.float32).to(device)
                y_real = torch.zeros((batch_size, 1), dtype=torch.float32).to(device)

                # zero the parameter gradients
                encoder.zero_grad()
                generator.zero_grad()
                discriminator.zero_grad()

                # DISCRIMINATOR TRAINING
                with torch.no_grad():
                    hidden, _ = encoder(in_frames)
                    generated, _ = generator(hidden)
                _, logits_real, _ = discriminator(out_frames)
                _, logits_gen, _ = discriminator(generated)
                # loss on real batch
                disc_loss_real = disc_loss_fun(logits_real, y_real)
                # loss on fake batch
                disc_loss_gen = disc_loss_fun(logits_gen, y_gen)

                # real-fake accuracy
                disc_acc_real = torch.mean((torch.sigmoid(logits_real.squeeze()) <= 0.5).float())
                disc_acc_gen = torch.mean((torch.sigmoid(logits_gen.squeeze()) > 0.5).float())

                disc_loss = (disc_loss_real + disc_loss_gen) / 2
                disc_loss.backward()

                discriminator_optim.step()

                # GENERATOR/ENCODER TRAINING
                encoder.zero_grad()
                generator.zero_grad()
                discriminator.zero_grad()

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
                disc_running_acc_real += disc_acc_real.item()
                disc_running_acc_gen += disc_acc_gen.item()
                if i % 10 == 9:
                    print('[Epoch {0} - Step {1}] Loss: (D) {2} - (G) {3} | Accuracy: (R) {4} - (G) {5}'.format(
                        epoch, i, round(disc_running_loss / 10, 4), round(gen_running_loss / 10, 4),
                        round(disc_running_acc_real / 10, 4), round(disc_running_acc_gen / 10, 4)))
                    tb_writer.add_scalar('D_loss_real', disc_running_loss / 10, global_step=global_step)
                    tb_writer.add_scalar('D_loss_gen', disc_running_loss / 10, global_step=global_step)
                    tb_writer.add_scalar('G_loss', gen_running_loss / 10, global_step=global_step)
                    tb_writer.add_scalar('D_acc_real', disc_running_acc_real / 10, global_step=global_step)
                    tb_writer.add_scalar('D_acc_gen', disc_running_acc_gen / 10, global_step=global_step)
                    # log generated images
                    gen_imgs = torchvision.utils.make_grid(generated.squeeze())
                    tb_writer.add_image('G_imgs', gen_imgs, global_step=global_step)

                    disc_running_loss = 0.0
                    gen_running_loss = 0.0
                    disc_running_acc_real = 0.0
                    disc_running_acc_gen = 0.0

                global_step += 1

    # write model to disk
    torch.save(encoder.state_dict(), os.path.join(args.logdir, 'encoder.pth'))
    torch.save(generator.state_dict(), os.path.join(args.logdir, 'generator.pth'))
    torch.save(discriminator.state_dict(), os.path.join(args.logdir, 'discriminator.pth'))
    torch.save({'train_idx': train_idx, 'val_idx': val_idx, 'test_idx': test_idx},
               os.path.join(args.logdir, 'data_idx.pth'))
