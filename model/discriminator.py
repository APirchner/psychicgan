import numpy as np
import torch
import torch.nn as nn

from model import layers


class Discrimator(nn.Module):
    def __init__(self, frame_dim=64, init_temp=3, feature_dim=128, filters=[64, 128, 256, 512], attention_at=8,
                 norm=nn.utils.weight_norm, residual=True):
        super(Discrimator, self).__init__()
        # check if spatial frame dim is power of 2
        assert not frame_dim & (frame_dim - 1) and not attention_at & (attention_at - 1)

        # go from 2^n up to 2^2
        self.depth = int(np.log2(frame_dim) - 2)

        # check if enough filters provided
        assert len(filters) == self.depth

        # get position of attention layer
        self.att_idx = self.depth - int(np.log2(attention_at)) + 2

        # starts out with RGB
        self.filters = [3]
        self.filters.extend(filters)

        temps = [True if i > 1 else False for i in range(init_temp, 0, -1)]
        temps = temps + [False for _ in range(self.depth - len(temps))]

        self.linear = nn.Linear(self.filters[-1], feature_dim, bias=True)
        self.logits = nn.Linear(feature_dim, 1, bias=True)

        self.down_stack = []

        for i in range(self.depth):
            if residual:
                self.down_stack.append(layers.ResidualNormConv3D(c_in=self.filters[i], c_out=self.filters[i + 1],
                                                                 activation_fun=nn.LeakyReLU(0.02),
                                                                 batchnorm=False,
                                                                 down_spatial=True, down_temporal=temps[i])
                                       )
            else:
                self.down_stack.append(layers.NormConv3D(c_in=self.filters[i], c_out=self.filters[i + 1],
                                                         activation_fun=nn.LeakyReLU(0.02),
                                                         batchnorm=False,
                                                         down_spatial=True, down_temporal=temps[i])
                                       )
        self.down_stack = nn.ModuleList(self.down_stack)

        self.attention = layers.SelfAttention3D(norm, c_in=self.filters[self.att_idx])

    def forward(self, input):
        attn = None
        x = input
        for i in range(len(self.down_stack)):
            # include attention layer at chosen depth
            if i == self.att_idx:
                x, attn = self.attention(x)
            x = self.down_stack[i](x)
        x = torch.mean(x, dim=[2, 3, 4])
        feat = self.linear(x)
        logits = self.logits(feat)
        return feat, logits.squeeze(), attn
