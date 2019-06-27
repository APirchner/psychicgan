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
    elif classname.find('Conv1d') != -1:
        nn.init.normal_(net.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
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
    parser.add_argument('-w', '--wasserstein', action='store_true', help='Use Wasserstein GAN')
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
    encoder = Encoder(frame_dim=64, init_temp=2, hidden_dim=args.latent_dim, filters=[16, 32, 64, 128],
                      attention_at=None, norm=nn.utils.spectral_norm, residual=True)
    encoder = encoder.to(device)
    encoder.apply(weight_init)

    # generator setup
    generator = Generator(frame_dim=64, temporal_target=1, hidden_dim=args.latent_dim,
                          filters=[256, 128, 64, 32], attention_at=32, norm=nn.utils.spectral_norm)
    generator = generator.to(device)
    generator.apply(weight_init)

    # discriminator setup
    discriminator = Discrimator(frame_dim=64, init_temp=1, feature_dim=128, filters=[16, 32, 64, 128],
                                attention_at=32, norm=nn.utils.spectral_norm, residual=True)
    discriminator = discriminator.to(device)
    discriminator.apply(weight_init)

    # print model summaries
    summary(encoder, input_size=(3, args.ins, 64, 64))
    summary(generator, input_size=(args.latent_dim,))
    summary(discriminator, input_size=(3, args.ins, 64, 64))

    # tensorboard log writer
    tb_writer = SummaryWriter(log_dir='/home/ewok261/Documents/psychic/sem_2/adl4cv/runs')

    if args.wasserstein:
        # WASSERSTEIN GAN

        # in WGAN, moment-based optimizers dont work as well -> see WGAN paper
        encoder_optim = optim.RMSprop(encoder.parameters(), lr=args.lr_encoder)
        generator_optim = optim.RMSprop(generator.parameters(), lr=args.lr_generator)
        discriminator_optim = optim.RMSprop(discriminator.parameters(), lr=args.lr_discriminator)

        encoder_sched = optim.lr_scheduler.ExponentialLR(encoder_optim, gamma=0.99)
        generator_sched = optim.lr_scheduler.ExponentialLR(generator_optim, gamma=0.99)
        discriminator_sched = optim.lr_scheduler.ExponentialLR(discriminator_optim, gamma=0.99)

        one = torch.FloatTensor([1]).to(device)
        m_one = -1 * one

        global_step = 0
        for epoch in range(args.epochs):
            i = 0
            train_iter = iter(train_loader)

            while i < len(train_loader):
                j = 0
                while j < 5 and i < len(train_loader) - 1:
                    # do 5 discriminator steps for each encoder/generator step -> see WGAN paper
                    # get the inputs and move them to device
                    data = next(train_iter)
                    in_frames, out_frames = data
                    in_frames = in_frames.to(device)
                    out_frames = out_frames.to(device)

                    # zero the parameter gradients
                    encoder_optim.zero_grad()
                    generator_optim.zero_grad()
                    discriminator.zero_grad()

                    # DISCRIMINATOR TRAINING
                    with torch.no_grad():
                        hidden, _ = encoder(in_frames)
                        generated, _ = generator(hidden)
                    _, out_real, _ = discriminator(out_frames)
                    _, out_gen, _ = discriminator(generated)

                    err_real = out_real.mean(0).view(1)
                    err_gen = out_gen.mean(0).view(1)

                    loss_D = err_gen - err_real
                    loss_D.backward()
                    #err_real.backward(m_one)
                    #err_gen.backward(one)
                    #err_D = err_real + err_gen
                    discriminator_optim.step()
                    j += 1
                    i += 1

                # GENERATOR/ENCODER TRAINING
                data = next(train_iter)
                in_frames, out_frames = data
                in_frames = in_frames.to(device)
                out_frames = out_frames.to(device)

                encoder_optim.zero_grad()
                generator_optim.zero_grad()
                discriminator.zero_grad()

                hidden, encoder_attn = encoder(in_frames)
                generated, generator_attn = generator(hidden)

                _, out_gen, _ = discriminator(generated)
                err_G = out_gen.mean(0).view(1)
                
                loss_G = -err_G
                loss_G.backward()
                #err_G.backward(one)
                generator_optim.step()
                encoder_optim.step()
                i += 1

                # print statistics
                if global_step % 10 == 9:
                    print('[Epoch {0} - Step {1}] Loss: (D) {2} - (G) {3}'.format(
                        epoch, global_step, round(err_D.item(), 4), round(err_G.item(), 4)))
                    tb_writer.add_scalar('D_loss', err_D.item(), global_step=global_step)
                    tb_writer.add_scalar('G_loss', err_G.item(), global_step=global_step)
                    # log generated images
                    gen_imgs = torchvision.utils.make_grid(generated.squeeze())
                    tb_writer.add_image('G_imgs', gen_imgs, global_step=global_step)

                global_step += 1


    else:
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
                encoder_optim.zero_grad()
                generator_optim.zero_grad()
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
                encoder_optim.zero_grad()
                generator_optim.zero_grad()
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
