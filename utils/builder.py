from args import args
import math

import torch
import torch.nn as nn

import utils.conv_type
import utils.bn_type

class Builder(object):
    def __init__(self, conv_layer, conv_rep_layer, rep_layer, bn_layer):
        self.conv_layer = conv_layer
        self.conv_rep_layer = conv_rep_layer
        self.rep_layer  = rep_layer
        self.bn_layer = bn_layer

    def conv_rep_conv(self, kernel_size, in_planes, out_planes, stride=1, groups=1):
        c = self.conv_rep_layer(in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=1,
                groups=groups,
                bias=False)
        self._init_conv(c)
        return c 
        
    def rep_conv(self, kernel_size, in_planes, out_planes, stride=1,groups=1):
        c = self.rep_layer(in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=1,
                groups=groups,
                bias=False)
        self._init_conv(c)
        return c

    def conv(self, kernel_size, in_planes, out_planes, stride=1, groups=1, bias=False):
        conv_layer = self.conv_layer

        if kernel_size == 3:
            conv = conv_layer(
                in_planes,
                out_planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=groups,
                bias=bias,
            )
        elif kernel_size == 1:
            conv = conv_layer(
                in_planes, out_planes, kernel_size=1, stride=stride, bias=bias
            )
        elif kernel_size == 5:
            conv = conv_layer(
                in_planes,
                out_planes,
                kernel_size=5,
                stride=stride,
                padding=2,
                bias=False,
            )
        elif kernel_size == 7:
            conv = conv_layer(
                in_planes,
                out_planes,
                kernel_size=7,
                stride=stride,
                padding=3,
                bias=False,
            )
        else:
            return None

        self._init_conv(conv)

        return conv

    def conv2d(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
    ):
        return self.conv_layer(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )

    def conv3x3(self, in_planes, out_planes, stride=1, bias=False, groups=1):
        """3x3 convolution with padding"""
        c = self.conv(3, in_planes, out_planes, stride=stride, groups=groups, bias=bias)
        return c

    def conv1x1(self, in_planes, out_planes, stride=1):
        """1x1 convolution with padding"""
        c = self.conv(1, in_planes, out_planes, stride=stride)
        return c
        
    def conv1x1_fc(self, in_planes, out_planes, stride=1):
        """full connect layer"""
        c = self.conv(1, in_planes, out_planes, stride=stride, bias=True)
        return c

    def conv7x7(self, in_planes, out_planes, stride=1):
        """7x7 convolution with padding"""
        c = self.conv(7, in_planes, out_planes, stride=stride)
        return c

    def conv5x5(self, in_planes, out_planes, stride=1):
        """5x5 convolution with padding"""
        c = self.conv(5, in_planes, out_planes, stride=stride)
        return c

    def batchnorm(self, planes, last_bn=False):
        return self.bn_layer(planes)

    def activation(self):
        if args.nonlinearity == "relu":
            return (lambda: nn.ReLU(inplace=True))()
        else:
            raise ValueError(f"{args.nonlinearity} is not an initialization option!")

    def _init_conv(self, conv):
        if args.init == "signed_constant":

            fan = nn.init._calculate_correct_fan(conv.weight, args.mode)
            if args.scale_fan:
                fan = fan * (1 - args.prune_rate)
            gain = nn.init.calculate_gain(args.nonlinearity)
            std = gain / math.sqrt(fan)
            conv.weight.data = conv.weight.data.sign() * std

        elif args.init == "unsigned_constant":

            fan = nn.init._calculate_correct_fan(conv.weight, args.mode)
            if args.scale_fan:
                fan = fan * (1 - args.prune_rate)

            gain = nn.init.calculate_gain(args.nonlinearity)
            std = gain / math.sqrt(fan)
            conv.weight.data = torch.ones_like(conv.weight.data) * std

        elif args.init == "kaiming_normal":

            if args.scale_fan:
                fan = nn.init._calculate_correct_fan(conv.weight, args.mode)
                fan = fan * (1 - args.prune_rate)
                gain = nn.init.calculate_gain(args.nonlinearity)
                std = gain / math.sqrt(fan)
                with torch.no_grad():
                    conv.weight.data.normal_(0, std)
            else:
                nn.init.kaiming_normal_(
                    conv.weight, mode=args.mode, nonlinearity=args.nonlinearity
                )

        elif args.init == "standard":
            nn.init.kaiming_uniform_(conv.weight, a=math.sqrt(5))
        else:
            raise ValueError(f"{args.init} is not an initialization option!")


def get_builder():

    print("==> Conv Type: {}".format(args.conv_type))
    print("==> BN Type: {}".format(args.bn_type))

    conv_layer = getattr(utils.conv_type, args.conv_type)
    conv_rep_conv = getattr(utils.conv_type, args.conv_rep_type)
    rep_conv = getattr(utils.conv_type, args.rep_type)
    bn_layer = getattr(utils.bn_type, args.bn_type)

    builder = Builder(conv_layer=conv_layer, conv_rep_layer=conv_rep_conv, rep_layer=rep_conv, bn_layer=bn_layer)

    return builder
