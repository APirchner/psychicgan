import numpy as np
import torch.nn as nn
import model.layers as layers


class Generator(nn.Module):
    def __init__(self, frame_dim=64, temporal_target=3, hidden_dim=128, init_filters=256, attention_at=8,
                 norm=nn.utils.weight_norm):
        super(Generator, self).__init__()
        # check if spatial frame dim is power of 2
        assert not frame_dim & (frame_dim - 1) and not attention_at & (attention_at - 1)

        # go from 2^2 up to 2^n
        self.depth = int(np.log2(frame_dim) - 2)
        # get position of attention layer
        self.att_idx = self.depth - int(np.log2(attention_at))
        self.init_filters = init_filters

        # get number of filters, initial number gets div by 2 every layer
        filters = [init_filters]
        filters.extend([init_filters // (2 ** i) for i in range(1, self.depth)])
        # last layer outputs 3 channels for RGB
        filters.append(3)

        # determine up-sampling sizes
        # spread temporal up-sampling over multiple layers
        temp_sizes = [i if i < temporal_target else temporal_target for i in range(2, self.depth + 2)]
        out_sizes = [(temp_sizes[i], 2 ** (i + 3), 2 ** (i + 3)) for i in range(self.depth)]

        self.linear = nn.Linear(hidden_dim, 4 * 4 * init_filters, bias=True)
        self.up_stack = [layers.NormUpsample3D(c_in=filters[i], c_out=filters[i + 1],
                                               out_size=out_sizes[i],
                                               activation_fun=nn.LeakyReLU if i < self.depth else nn.Tanh
                                               ) for i in range(self.depth)]
        self.attention = layers.SelfAttention3D(norm, c_in=filters[self.att_idx])

    def forward(self, *input):
        x = self.linear(input)
        x = x.reshape((-1, self.init_filters, 1, 4, 4))
        for i in range(len(self.up_stack)):
            # include attention layer at chosen depth
            if i == self.att_idx:
                x, attn = self.attention(x)
            x = self.up_stack[i](x)
        return x, attn
