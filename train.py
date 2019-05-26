import argparse
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim

from utils.data_utils import *
from model.encoder import Encoder
from model.generator import Generator

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # data set args
    parser.add_argument('-d', '--dir', type=str, required=True, help='The data directory')
    parser.add_argument('-i', '--ins', type=int, required=True, help='The number of conditioning frames')
    parser.add_argument('-o', '--outs', type=int, required=True, help='The number of generated frames')
    # optimizer args
    parser.add_argument('-e', '--epochs', type=int, default=20, help='The number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='The batch size')

    parser.add_argument('-c', '--disable-cuda', action='store_false', help='Disable CUDA')
    args = parser.parse_args()

    device = None
    if not args.disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    train_data = KITTIData(args.ins, args.outs, 0, args.dir, )
    train_loader = data.DataLoader(train_data, batch_size=1, shuffle=True, num_workers=1)

    loss_fun = nn.MSELoss()

    encoder = Encoder(frame_dim=64, init_temp=2, hidden_dim=128, out_filters=256,
                      attention_at=8, norm=nn.utils.weight_norm, residual=True).to(device)
    encoder_optim = optim.Adam(encoder.parameters())

    generator = Generator(frame_dim=64, temporal_target=1, hidden_dim=128,
                          init_filters=256, attention_at=8, norm=nn.utils.weight_norm).to(device)
    generator_optim = optim.Adam(generator.parameters())

    for epoch in range(args.epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            in_frames, out_frames = data
            in_frames = in_frames.to(device)

            # zero the parameter gradients
            encoder_optim.zero_grad()
            generator_optim.zero_grad()

            # forward + backward + optimize
            hidden, encoder_attn = encoder(in_frames)
            generated, generator_attn = generator(hidden)
            loss = loss_fun(generated, out_frames)

            loss.backward()
            generator_optim.step()
            encoder_optim.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:
                print('[Epoch {0} - Step {1}] Loss: {2}'.format(epoch, i, loss / 10))
