import numpy as np
import torch.nn as nn

import model.layers as layers


class Encoder(nn.Module):
    def __init__(self, frame_dim=64, init_temp=3, hidden_dim=128, out_filters=256, attention_at=8,
                 norm=nn.utils.weight_norm, residual=True):
        super(Encoder, self).__init__()
        # check if spatial frame dim is power of 2
        assert not frame_dim & (frame_dim - 1) and not attention_at & (attention_at - 1)

        # go from 2^n up to 2^2
        self.depth = int(np.log2(frame_dim) - 2)
        # get position of attention layer
        self.att_idx = int(np.log2(attention_at))
        self.out_filters = out_filters

        # get number of filters, target number gets div by 2 every layer
        filters = [3]
        filters.extend([self.out_filters // (2 ** i) for i in range(self.depth-1, 0, -1)])
        filters.append(self.out_filters)

        temps = [True if i > 1 else False for i in range(init_temp, 0, -1)]
        temps = temps + [False for i in range(self.depth - len(temps))]

        self.linear = nn.Linear(4 * 4 * out_filters, hidden_dim, bias=True)

        if residual:
            self.down_stack = nn.ModuleList([layers.ResidualNormConv3D(c_in=filters[i], c_out=filters[i + 1],
                                                                       activation_fun=nn.LeakyReLU,
                                                                       down_spatial=True, down_temporal=temps[i]
                                                                       ) for i in range(self.depth)])
        else:
            self.down_stack = nn.ModuleList([layers.NormConv3D(c_in=filters[i], c_out=filters[i + 1],
                                                               activation_fun=nn.LeakyReLU,
                                                               down_spatial=True, down_temporal=temps[i]
                                                               ) for i in range(self.depth)])
        self.attention = layers.SelfAttention3D(norm, c_in=filters[self.att_idx])

    def forward(self, input):
        attn = None
        x = input
        for i in range(len(self.down_stack)):
            # include attention layer at chosen depth
            if i == self.att_idx:
                x, attn = self.attention(x)
            x = self.down_stack[i](x)
        x = x.reshape((-1, 4 * 4 * self.out_filters))
        x = self.linear(x)
        return x, attn


if __name__ == '__main__':
    Encoder()
