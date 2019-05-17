import torch
import torch.nn as nn


class NormConvND(nn.Module):
    """ Wrapper for N-dimensional convolution with normalization """

    def __init__(self, conv, c_in, c_out, kernel_size, stride, bias=True,
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
        self.conv = conv(in_channels=c_in, out_channels=c_out, kernel_size=kernel_size,
                         stride=stride, bias=bias, padding=padding)
        self.norm = norm
        self.activation_fun = activation_fun

    def forward(self, *input):
        x = self.norm(self.conv(input)) if self.norm is not None else self.conv(input)
        x = self.activation_fun(x) if self.activation_fun is not None else x
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
        self.trans_conv = trans_conv(in_channels=c_in, out_channels=c_out, kernel_size=kernel_size,
                                     stride=stride, bias=bias, padding=padding, output_padding=output_padding)
        self.norm = norm
        self.activation_fun = activation_fun

    def forward(self, *input):
        x = self.norm(self.trans_conv(input)) if self.norm is not None else self.trans_conv(input)
        x = self.activation_fun(x) if self.activation_fun is not None else x
        return x


class SelfAttentionND(nn.Module):
    """ N-dimensional self attention layer. N is 2 or 3. """

    def __init__(self, dim, norm, c_in):
        """
        :param dim: input dimension, either 2 or 3
        :param norm: the kind of normalization for the convolutions, e.g. torch.utils.weight_norm
        :param c_in: the input channels
        """
        super(SelfAttentionND, self).__init__()

        assert dim == 2 or dim == 3

        # channel reduction like in self-attention GAN paper
        self.c_inter = c_in // 8
        self.query_conv = NormConvND(conv=nn.Conv3d if dim == 3 else nn.Conv2d, c_in=c_in, c_out=self.c_inter,
                                     kernel_size=1, stride=1, bias=True, norm=norm)
        self.key_conv = NormConvND(conv=nn.Conv3d if dim == 3 else nn.Conv2d, c_in=c_in, c_out=self.c_inter,
                                   kernel_size=1, stride=1, bias=True, norm=norm)
        self.value_conv = NormConvND(conv=nn.Conv3d if dim == 3 else nn.Conv2d, c_in=c_in, c_out=self.c_inter,
                                     kernel_size=1, stride=1, bias=True, norm=norm)
        # conv to restore number of input channels
        self.att_conv = NormConvND(conv=nn.Conv3d if dim == 3 else nn.Conv2d, c_in=self.c_inter, c_out=c_in,
                                   kernel_size=1, stride=1, bias=True, norm=norm)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, *input):
        """

        :param input: layer input
        :return: tuple of layer output and attention mask
        """
        # get batch size to infer other dims
        input_size = input.size()

        # split input and reduce channels [batch, c, (t), h, w]
        query = self.query_conv(input)
        key = self.key_conv(input)
        value = self.value_conv(input)

        # reshape to [batch, c_inter, (t) * h * w]
        query = query.view(input_size[0], self.c_inter, -1)
        key = key.view(input_size[0], self.c_inter, -1)
        value = value.view(input_size[0], self.c_inter, -1)

        # transpose to [batch, (t) * h * w, c_inter]
        query_t = query.permute(0, 2, 1)
        value_t = value.permute(0, 2, 1)

        # query key product and softmax
        # [batch, (t) * h * w, c_inter] * [batch, c_inter, (t) * h * w]
        query_x_key = torch.bmm(query_t, key)
        query_x_key = self.softmax(query_x_key)

        # query-key value product
        res = torch.bmm(query_x_key, value_t)
        res = res.view(input_size[0], self.c_inter, *input_size[2:])
        res = self.att_conv(res)

        out = input + res
        return out, res
