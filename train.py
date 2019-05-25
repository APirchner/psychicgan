import numpy as np
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim

from utils.data_utils import *
from model.encoder import Encoder
from model.generator import Generator

if __name__ == '__main__':

    train_data = KITTIData(2, 1, 0,
                           '/home/andreas/Documents/msc_info/sem_2/adl4cv/2011_09_28_drive_0053_sync/in_2_out_1_ol_0')
    train_loader = data.DataLoader(train_data, batch_size=1, shuffle=True, num_workers=1)

    loss_fun = nn.MSELoss()

    encoder = Encoder(frame_dim=64, init_temp=2, hidden_dim=128, out_filters=256,
                      attention_at=8, norm=nn.utils.weight_norm, residual=True)
    encoder_optim = optim.Adam(encoder.parameters())

    generator = Generator(frame_dim=64, temporal_target=1, hidden_dim=128,
                          init_filters=256, attention_at=8, norm=nn.utils.weight_norm)
    generator_optim = optim.Adam(generator.parameters())

    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            in_frames, out_frames = data

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
            if i % 2 == 1:  # print every 2 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
