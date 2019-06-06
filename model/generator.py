import numpy as np
import torch.nn as nn
import model.layers as layers


class Generator(nn.Module):
    def __init__(self, frame_dim=64, temporal_target=3, hidden_dim=128, filters=[512, 256, 128, 64], attention_at=8,
                 norm=nn.utils.weight_norm):
        super(Generator, self).__init__()
        # check if spatial frame dim is power of 2
        assert not frame_dim & (frame_dim - 1)

        # go from 2^2 up to 2^n
        self.depth = int(np.log2(frame_dim)) - 2

        # check if enough filters provided
        assert len(filters) == self.depth

        # get position of attention layer
        self.att_idx = int(np.log2(attention_at)) - 2 if attention_at is not None else None

        self.filters = filters
        # last layer outputs 3 channels for RGB
        self.filters.append(3)

        # determine up-sampling sizes
        # spread temporal up-sampling over multiple layers
        temp_sizes = [i if i < temporal_target else temporal_target for i in range(2, self.depth + 2)]
        out_sizes = [(temp_sizes[i], 2 ** (i + 3), 2 ** (i + 3)) for i in range(self.depth)]

        self.linear = nn.Linear(hidden_dim, 4 * 4 * self.filters[0], bias=True)

        self.up_stack = []

        for i in range(self.depth):
            self.up_stack.append(layers.NormUpsample3D(
                c_in=self.filters[i], c_out=self.filters[i + 1],
                out_size=out_sizes[i],
                batchnorm=True,
                activation_fun=nn.LeakyReLU(0.2) if i < self.depth - 1 else nn.Tanh()
            )
            )
        self.up_stack = nn.ModuleList(self.up_stack)

        self.attention = layers.SelfAttention3D(norm, c_in=self.filters[self.att_idx]) \
            if attention_at is not None else None

    def forward(self, input):
        attn = None
        x = self.linear(input)
        x = x.reshape((-1, self.filters[0], 1, 4, 4))
        for i in range(len(self.up_stack)):
            # include attention layer at chosen depth
            if self.att_idx is not None and i == self.att_idx:
                x, attn = self.attention(x)
            x = self.up_stack[i](x)
        return x, attn
