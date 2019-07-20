import argparse
import os
import json

import torch
import torch.utils.data as data
import torch.nn as nn

from torchsummary import summary
from torchvision.utils import save_image

from utils.data_utils import FramesData
from model.encoder import Encoder, EncoderMoreConvs
from model.generator import Generator, GeneratorMoreConvs
from model.discriminator import Discrimator, DiscrimatorMoreConvs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data set args
    parser.add_argument('-d', '--dir', type=str, required=True, help='The data directory')
    parser.add_argument('-l', '--logdir', type=str, required=True, help='The log directory')
    # CUDA
    parser.add_argument('-c', '--disable-cuda', action='store_true', help='Disable CUDA')

    args = parser.parse_args()

    # check if logdir exists
    if not os.path.exists(args.logdir):
        raise ValueError('Logdir you\'re trying to load models from does not exist!')

    with open(os.path.join(args.logdir, 'args.txt'), 'r') as f:
        train_args = json.load(f)

    train_ins = train_args['ins']
    train_outs = train_args['outs']
    train_lat_dim = train_args['latent_dim']
    train_config = train_args['config']
    train_wgan = train_args['wasserstein']

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

    train_loader = data.DataLoader(train_data, batch_size=1, shuffle=True)
    # val_loader = data.DataLoader(val_data, batch_size=32, shuffle=False, num_workers=args.workers)
    test_loader = data.DataLoader(test_data, batch_size=1, shuffle=True)

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
            encoder = EncoderMoreConvs(frame_dim=64, init_temp=train_ins, hidden_dim=train_lat_dim,
                                       filters=[16, 32, 64, 128], attention_at=16, target_temp=1, norm=None,
                                       batchnorm=True, dropout=0.0, residual=True)
            generator = GeneratorMoreConvs(frame_dim=64, temporal_target=train_outs, hidden_dim=train_lat_dim,
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

    generator.train(False)
    encoder.train(False)
    discriminator.train(False)

    # print model summaries
    summary(encoder, input_size=(3, train_ins, 64, 64))
    summary(generator, input_size=(train_lat_dim,))
    summary(discriminator, input_size=(3, 1+train_outs, 64, 64))

    nims = 10

    for in_frames, out_frames in train_loader:
        in_frames = in_frames.to(device)
        out_frames = out_frames.to(device)

        with torch.no_grad():
            # forward
            hidden, _ = encoder(in_frames)
            generated, _ = generator(hidden)

            out_frames_disc = torch.cat([in_frames, out_frames], dim=2).squeeze().permute(1, 0, 2, 3)
            generated_disc = torch.cat([in_frames, generated], dim=2).squeeze().permute(1, 0, 2, 3)

        save_image(torch.cat([(out_frames_disc+1)/2, (generated_disc+1)/2], dim=0), filename=os.path.join(args.logdir, 'train'+str(nims)+'.png'),
                   nrow=train_ins+train_outs)
        nims -= 1
        if nims == 0:
            break

    nims = 10

    for in_frames, out_frames in test_loader:
        in_frames = in_frames.to(device)
        out_frames = out_frames.to(device)

        with torch.no_grad():
            # forward
            hidden, _ = encoder(in_frames)
            generated, _ = generator(hidden)

            out_frames_disc = torch.cat([in_frames, out_frames], dim=2).squeeze().permute(1, 0, 2, 3)
            generated_disc = torch.cat([in_frames, generated], dim=2).squeeze().permute(1, 0, 2, 3)

        save_image(torch.cat([(out_frames_disc+1)/2, (generated_disc+1)/2], dim=0), filename=os.path.join(args.logdir, 'test'+str(nims)+'.png'),
                   nrow=train_ins+train_outs)
        nims -= 1
        if nims == 0:
            break
