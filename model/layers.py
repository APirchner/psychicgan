import torch
import torch.nn as nn
import torch.nn.functional as F


class NormConvND(nn.Module):
    """ Wrapper for N-dimensional convolution with normalization """

    def __init__(self, conv, c_in, c_out, kernel_size, stride, bias=True, batchnorm=None,
                 norm=nn.utils.weight_norm, activation_fun=None, padding=0):
        """
        :param conv: the convolution function, e.g. torch.nn.Conv2D
        :param c_in: the input channels
        :param c_out: the output channels
        :param kernel_size: the kernel size, should be odd
        :param stride: the stride
        :param bias: use bias?
        :param norm: the normalization function for the weights, e.g. torch.utils.weight_norm
        :param activation_fun: the activation function, e.g. ReLU
        :param padding: the amount of zero padding
        """
        super(NormConvND, self).__init__()
        self.container = nn.ModuleDict({
            'conv': norm(conv(in_channels=c_in, out_channels=c_out, kernel_size=kernel_size,
                              stride=stride, bias=bias, padding=padding), name='weight'
                         ) if norm is not None else conv(in_channels=c_in, out_channels=c_out, kernel_size=kernel_size,
                                                         stride=stride, bias=bias, padding=padding
                                                         ),
            'bn': batchnorm,
            'activation': activation_fun
        })

    def forward(self, input):
        x = self.container['bn'](self.container['conv'](input)) \
            if self.container['bn'] is not None else self.container['conv'](input)
        x = self.container['activation'](x) if self.container['activation'] is not None else x
        return x


class NormTransConvND(nn.Module):
    """ Wrapper for N-dimensional transposed convolution with normalization """

    def __init__(self, trans_conv, c_in, c_out, kernel_size, stride, bias=False,
                 norm=nn.utils.weight_norm, activation_fun=None, padding=0, output_padding=0):
        """
        :param conv: the convolution function, e.g. torch.nn.Conv2D
        :param c_in: the input channels
        :param c_out: the output channels
        :param kernel_size: the kernel size, should be odd
        :param stride: the stride
        :param bias: use bias?
        :param norm: the normalization function for the weights, e.g. torch.utils.weight_norm
        :param activation_fun: the activation function, e.g. ReLU
        :param padding: the amount of zero padding
        :param output_padding: the padding to add to the output
        """
        super(NormTransConvND, self).__init__()
        self.container = nn.ModuleDict({
            'conv': norm(trans_conv(
                in_channels=c_in, out_channels=c_out, kernel_size=kernel_size,
                stride=stride, bias=bias, padding=padding,
                output_padding=output_padding)
            ) if norm is not None else trans_conv(in_channels=c_in, out_channels=c_out,
                                                  kernel_size=kernel_size,
                                                  stride=stride, bias=bias,
                                                  padding=padding,
                                                  output_padding=output_padding),
            'activation': activation_fun if activation_fun is not None else None
        })

    def forward(self, input):
        x = self.container['conv'](input)
        x = self.container['activation'](x) if self.container['activation'] is not None else x
        return x


class NormUpsampleND(nn.Module):
    """
     Wrapper for an N-dimensional up-sampling followed by a convolution.
     Should be a substitute for a transposed convolution.
    """

    def __init__(self, conv, c_in, c_out, out_size, bias=True, batchnorm=None,
                 activation_fun=None, norm=nn.utils.weight_norm, mode='nearest'):
        """
        :param conv: the convolution function, e.g. torch.nn.Conv2D
        :param c_in: the input channels
        :param c_out: the output channels
        :param out_size: the tensor size of the output of up-sampling [c, (t), h, w]
        :param kernel_size: the kernel size of the conv after up-sampling
        :param activation_fun: the activation function e.g. ReLU
        :param bias: use bias?
        :param norm: the normalization function for the weights, e.g. torch.utils.weight_norm
        :param padding: the amount of zero padding in the convolution
        :param mode: the mode of the upsampling, 'nearest', 'linear', etc.
        """
        super(NormUpsampleND, self).__init__()
        self.container = nn.ModuleDict({
            'upsample': nn.Upsample(size=out_size, mode=mode),
            'conv': norm(conv(in_channels=c_in, out_channels=c_out, kernel_size=3,
                              stride=1, bias=False, padding=1)
                         ) if norm is not None else conv(in_channels=c_in, out_channels=c_out, kernel_size=3,
                                                         stride=1, bias=bias, padding=1),
            'bn': batchnorm,
            'activation': activation_fun
        })

    def forward(self, input):
        x = self.container['upsample'](input)
        x = self.container['bn'](self.container['conv'](x)) \
            if self.container['bn'] is not None else self.container['conv'](x)
        x = self.container['activation'](x) if self.container['activation'] is not None else x
        return x


class SelfAttentionND(nn.Module):
    """ N-dimensional self attention layer. N is 2 or 3. """

    def __init__(self, dim, norm, c_in, bias=True):
        """
        :param dim: input dimension, either 2 or 3
        :param norm: the kind of normalization for the convolutions, e.g. torch.utils.weight_norm
        :param c_in: the input channels
        """
        super(SelfAttentionND, self).__init__()

        assert dim == 2 or dim == 3

        # channel reduction like in self-attention GAN paper
        self.c_inter = c_in // 8
        self.c_inter_value = c_in // 2

        # trainable attention mix parameter
        self.gamma = nn.Parameter(data=torch.zeros(1, dtype=torch.float32))

        self.container = nn.ModuleDict({
            'query_conv': NormConvND(conv=nn.Conv3d if dim == 3 else nn.Conv2d, c_in=c_in, c_out=self.c_inter,
                                     kernel_size=1, stride=1, bias=bias, norm=norm),
            'key_conv': NormConvND(conv=nn.Conv3d if dim == 3 else nn.Conv2d, c_in=c_in, c_out=self.c_inter,
                                   kernel_size=1, stride=1, bias=bias, norm=norm),
            'value_conv': NormConvND(conv=nn.Conv3d if dim == 3 else nn.Conv2d, c_in=c_in, c_out=self.c_inter_value,
                                     kernel_size=1, stride=1, bias=bias, norm=norm),
            'att_conv': NormConvND(conv=nn.Conv3d if dim == 3 else nn.Conv2d, c_in=self.c_inter_value, c_out=c_in,
                                   kernel_size=1, stride=1, bias=bias, norm=norm),
            'softmax': nn.Softmax(dim=-1)
        })

    def forward(self, input):
        # get batch size to infer other dims
        input_size = list(input.shape)

        # split input and reduce channels [batch, c, (t), h, w]
        query = self.container['query_conv'](input)
        key = self.container['key_conv'](input)
        value = self.container['value_conv'](input)

        # reshape to [batch, c_inter, (t) * h * w]
        query = query.view(input_size[0], self.c_inter, -1)
        key = key.view(input_size[0], self.c_inter, -1)
        value = value.view(input_size[0], self.c_inter_value, -1)

        # transpose to [batch, (t) * h * w, c_inter]
        query_t = query.permute(0, 2, 1)
        value_t = value.permute(0, 2, 1)

        # query key product and softmax
        # [batch, (t) * h * w, c_inter] * [batch, c_inter, (t) * h * w]
        query_x_key = torch.bmm(query_t, key)
        query_x_key = self.container['softmax'](query_x_key)
        attn_out = query_x_key.view(input_size[0], *input_size[2:], *input_size[2:])

        # query-key value product
        res = torch.bmm(query_x_key, value_t)
        res = res.view(input_size[0], self.c_inter_value, *input_size[2:])
        res = self.container['att_conv'](res)

        out = input + self.gamma * res
        return out, attn_out


class DummySelfAttentionND(nn.Module):
    """
    N-dimensional dummy self attention layer. N is 2 or 3.
    CAUTION: does not compute self attention, just has the same number of parameters
             to mimic the effect of the parameter count of the self attention layer!
    """

    def __init__(self, dim, norm, c_in, bias=True):
        """
        :param dim: input dimension, either 2 or 3
        :param norm: the kind of normalization for the convolutions, e.g. torch.utils.weight_norm
        :param c_in: the input channels
        """
        super(DummySelfAttentionND, self).__init__()

        assert dim == 2 or dim == 3

        # channel reduction like in self-attention GAN paper
        self.c_inter = c_in // 8
        self.c_inter_value = c_in // 2

        # trainable attention mix parameter
        self.gamma = nn.Parameter(data=torch.zeros(1, dtype=torch.float32))

        self.container = nn.ModuleDict({
            'query_conv': NormConvND(conv=nn.Conv3d if dim == 3 else nn.Conv2d, c_in=c_in, c_out=self.c_inter,
                                     kernel_size=1, stride=1, bias=bias, norm=norm),
            'key_conv': NormConvND(conv=nn.Conv3d if dim == 3 else nn.Conv2d, c_in=c_in, c_out=self.c_inter,
                                   kernel_size=1, stride=1, bias=bias, norm=norm),
            'value_conv': NormConvND(conv=nn.Conv3d if dim == 3 else nn.Conv2d, c_in=c_in, c_out=self.c_inter_value,
                                     kernel_size=1, stride=1, bias=bias, norm=norm),
            'att_conv': NormConvND(conv=nn.Conv3d if dim == 3 else nn.Conv2d,
                                   c_in=2 * self.c_inter + self.c_inter_value, c_out=c_in,
                                   kernel_size=1, stride=1, bias=bias, norm=norm),
        })

    def forward(self, input):
        """
        :param input: layer input
        :return: tuple of layer output and attention mask
        """
        # get batch size to infer other dims
        input_size = list(input.shape)

        # split input and reduce channels [batch, c, (t), h, w]
        query = self.container['query_conv'](input)
        key = self.container['key_conv'](input)
        value = self.container['value_conv'](input)

        res = torch.cat((query, key, value), dim=1)

        res = self.container['att_conv'](res)

        out = input + self.gamma * res
        return out, res


class NormConv3D(nn.Module):
    """
    3D convenience wrapper for ND conv with 3x3x3 kernel
    """

    def __init__(self, c_in, c_out, down_spatial=True, down_temporal=True, bias=True, batchnorm=True,
                 norm=nn.utils.weight_norm, activation_fun=None):
        super(NormConv3D, self).__init__()

        if down_spatial and down_temporal:
            stride = (2, 2, 2)
        elif down_spatial and not down_temporal:
            stride = (1, 2, 2)
        elif not down_spatial and down_temporal:
            stride = (2, 1, 1)
        else:
            stride = (1, 1, 1)

        self.layer = NormConvND(nn.Conv3d, c_in, c_out, 3, stride,
                                bias, nn.BatchNorm3d(c_out) if batchnorm else None, norm, activation_fun, 1)

    def forward(self, input):
        x = self.layer(input)
        return x


class ResidualNormConv3D(nn.Module):
    """
    3D convenience wrapper for ND conv with 3x3x3 kernel
    """

    def __init__(self, c_in, c_out, down_spatial=True, down_temporal=True, bias=True, batchnorm=True,
                 norm=nn.utils.weight_norm, activation_fun=None):
        super(ResidualNormConv3D, self).__init__()

        self.down_spatial = down_spatial
        self.down_temporal = down_temporal
        self.c_in = c_in

        if down_spatial and down_temporal:
            stride = (2, 2, 2)
        elif down_spatial and not down_temporal:
            stride = (1, 2, 2)
        elif not down_spatial and down_temporal:
            stride = (2, 1, 1)
        else:
            stride = (1, 1, 1)

        # strided 3x3 convolution
        self.layer = NormConvND(nn.Conv3d, c_in, c_out, 3, stride,
                                bias, nn.BatchNorm3d(c_out) if batchnorm else None, norm, activation_fun, 1)
        # residual pass 1x1 convolution to match filter number
        self.layer_res = NormConvND(nn.Conv3d, c_in, c_out, 1, 1,
                                    bias, nn.BatchNorm3d(c_out) if batchnorm else None, norm, None, 0)

    def forward(self, input):
        in_shape = list(input.shape)

        if self.down_spatial and self.down_temporal:
            out_size = (in_shape[2] // 2, in_shape[3] // 2, in_shape[4] // 2)
        elif self.down_spatial and not self.down_temporal:
            out_size = (in_shape[2], in_shape[3] // 2, in_shape[4] // 2)
        elif not self.down_spatial and self.down_temporal:
            out_size = (in_shape[2] // 2, in_shape[3], in_shape[4])
        else:
            out_size = in_shape[2:]

        x = self.layer(input) + self.layer_res(F.interpolate(input, out_size))
        return x


class NormConv2D(nn.Module):
    """
    3D convenience wrapper for ND conv
    """

    def __init__(self, c_in, c_out, kernel_size, stride, bias=True, norm=nn.utils.weight_norm, activation_fun=None):
        super(NormConv2D, self).__init__()
        self.layer = NormConvND(nn.Conv2d, c_in, c_out, kernel_size, stride,
                                bias, norm, activation_fun, padding=kernel_size // 2)

    def forward(self, input):
        x = self.layer(input)
        return x


class ResidualNormConv2D(nn.Module):
    """
    3D convenience wrapper for ND conv with 3x3x3 kernel
    """

    def __init__(self, c_in, c_out, down_spatial=True, bias=True, batchnorm=True,
                 norm=nn.utils.weight_norm, activation_fun=None):
        super(ResidualNormConv2D, self).__init__()

        self.down_spatial = down_spatial
        self.c_in = c_in

        if down_spatial:
            stride = (1, 2, 2)
        else:
            stride = (1, 1, 1)

        # strided 3x3 convolution
        self.layer = NormConvND(nn.Conv3d, c_in, c_out, (1, 3, 3), stride,
                                bias, nn.BatchNorm3d(c_out) if batchnorm else None, norm, activation_fun, 1)
        # residual pass 1x1 convolution to match filter number
        self.layer_res = NormConvND(nn.Conv3d, c_in, c_out, 1, 1,
                                    bias, nn.BatchNorm3d(c_out) if batchnorm else None, norm, None, 0)

    def forward(self, input):
        in_shape = list(input.shape)

        if self.down_spatial:
            out_size = (in_shape[2], in_shape[3] // 2, in_shape[4] // 2)
        else:
            out_size = in_shape[2:]

        x = self.layer(input) + self.layer_res(F.interpolate(input, out_size))
        return x


class DoubleConvBlock3D(nn.Module):
    """
    convenience wrapper for 2 ND convs
    """

    def __init__(self, c_in, c_out, down_spatial=True, down_temporal=True, bias=True, batchnorm=True,
                 norm=nn.utils.weight_norm, activation_fun=None):
        super(DoubleConvBlock3D, self).__init__()

        if down_spatial and down_temporal:
            stride = (2, 2, 2)
        elif down_spatial and not down_temporal:
            stride = (1, 2, 2)
        elif not down_spatial and down_temporal:
            stride = (2, 1, 1)
        else:
            stride = (1, 1, 1)

        # strided 3x3 convolutions
        self.conv1 = NormConvND(nn.Conv3d, c_in, c_in, 3, 1,
                                bias, nn.BatchNorm3d(c_in) if batchnorm else None, norm, activation_fun, 1)
        self.conv2 = NormConvND(nn.Conv3d, c_in, c_out, 3, stride,
                                bias, nn.BatchNorm3d(c_out) if batchnorm else None, norm, activation_fun, 1)

    def forward(self, input):
        x = self.conv2(self.conv1(input))
        return x


class ResidualDoubleConvBlock3D(nn.Module):
    """
    convenience wrapper for 2 ND convs
    """

    def __init__(self, c_in, c_out, down_spatial=True, down_temporal=True, bias=True, batchnorm=True,
                 norm=nn.utils.weight_norm, activation_fun=None):
        super(ResidualDoubleConvBlock3D, self).__init__()

        self.down_spatial = down_spatial
        self.down_temporal = down_temporal

        if down_spatial and down_temporal:
            stride = (2, 2, 2)
        elif down_spatial and not down_temporal:
            stride = (1, 2, 2)
        elif not down_spatial and down_temporal:
            stride = (2, 1, 1)
        else:
            stride = (1, 1, 1)

        # strided 3x3 convolutions
        self.conv1 = NormConvND(nn.Conv3d, c_in, c_in, 3, 1,
                                bias, nn.BatchNorm3d(c_in) if batchnorm else None, norm, activation_fun, 1)
        self.conv2 = NormConvND(nn.Conv3d, c_in, c_out, 3, stride,
                                bias, nn.BatchNorm3d(c_out) if batchnorm else None, norm, activation_fun, 1)

        # residual pass 1x1 convolution to match filter number
        self.layer_res = NormConvND(nn.Conv3d, c_in, c_out, 1, 1,
                                    bias, nn.BatchNorm3d(c_out) if batchnorm else None, norm, None, 0)

    def forward(self, input):
        in_shape = list(input.shape)

        if self.down_spatial and self.down_temporal:
            out_size = (in_shape[2] // 2, in_shape[3] // 2, in_shape[4] // 2)
        elif self.down_spatial and not self.down_temporal:
            out_size = (in_shape[2], in_shape[3] // 2, in_shape[4] // 2)
        elif not self.down_spatial and self.down_temporal:
            out_size = (in_shape[2] // 2, in_shape[3], in_shape[4])
        else:
            out_size = in_shape[2:]

        x = self.conv2(self.conv1(input))
        x += self.layer_res(F.interpolate(input, out_size))
        return x


class NormUpsample3D(nn.Module):
    """
    3D convenience wrapper for ND up-sample
    """

    def __init__(self, c_in, c_out, out_size, bias=True, batchnorm=True, activation_fun=None, norm=nn.utils.weight_norm,
                 mode='trilinear'):
        super(NormUpsample3D, self).__init__()
        self.layer = NormUpsampleND(nn.Conv3d, c_in, c_out, out_size, bias,
                                    nn.BatchNorm3d(c_out) if batchnorm else None, activation_fun, norm, mode)

    def forward(self, input):
        x = self.layer(input)
        return x


class NormUpsample2D(nn.Module):
    """
    2D convenience wrapper for ND up-sample
    """

    def __init__(self, c_in, c_out, out_size, bias=True, batchnorm=True, activation_fun=None, norm=nn.utils.weight_norm,
                 mode='bilinear'):
        super(NormUpsample2D, self).__init__()
        self.layer = NormUpsampleND(nn.Conv2d, c_in, c_out, out_size, bias,
                                    nn.BatchNorm3d(c_out) if batchnorm else None, activation_fun, norm, mode)

    def forward(self, input):
        x = self.layer(input)
        return x


class SelfAttention3D(nn.Module):
    """
    3D convenience wrapper for ND self-attention
    """

    def __init__(self, norm, c_in):
        super(SelfAttention3D, self).__init__()
        self.layer = SelfAttentionND(3, norm, c_in)

    def forward(self, input):
        x, attention_mask = self.layer(input)
        return x, attention_mask


class DummySelfAttention3D(nn.Module):
    """
    3D convenience wrapper for ND self-attention
    """

    def __init__(self, norm, c_in):
        super(DummySelfAttention3D, self).__init__()
        self.layer = DummySelfAttentionND(3, norm, c_in)

    def forward(self, input):
        x, attention_mask = self.layer(input)
        return x, attention_mask


class SelfAttention2D(nn.Module):
    """
    2D convenience wrapper for ND self-attention
    """

    def __init__(self, norm, c_in):
        super(SelfAttention2D, self).__init__()
        self.layer = SelfAttentionND(2, norm, c_in)

    def forward(self, input):
        x, attention_mask = self.layer(input)
        return x, attention_mask


class NormLinear(nn.Module):
    """
    Convenience wrapper for  a normalized linear layer
    """

    def __init__(self, c_in, c_out, activation_fun=None, batchnorm=True,
                 norm=nn.utils.spectral_norm, bias=True):
        super(NormLinear, self).__init__()
        self.container = nn.ModuleDict({
            'linear': norm(
                nn.Linear(in_features=c_in, out_features=c_out, bias=bias)
            ) if norm is not None else nn.Linear(in_features=c_in, out_features=c_out, bias=bias),
            'batchnorm': nn.BatchNorm1d(c_out) if batchnorm else None,
            'activation': activation_fun
        })

    def forward(self, input):
        x = self.container['linear'](input)
        x = self.container['batchnorm'](x) if self.container['batchnorm'] is not None else x
        x = self.container['activation'](x) if self.container['activation'] is not None else x
        return x
