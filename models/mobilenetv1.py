import torch.nn as nn
from utils.builder import get_builder

class conv_dw(nn.Module):
    def __init__(self, builder, inp, oup, stride, deploy=False):
        super(conv_dw, self).__init__()
        self.deploy = deploy
        if self.deploy:
            self.conv_deploy = builder.conv(3, inp, inp, stride, groups=inp, bias=True)
        else:
            self.conv1 = builder.conv_rep_conv(3, inp, inp, stride, groups=inp)
            self.bn1 = builder.batchnorm(inp)
            self.conv_rep = builder.rep_conv(3, inp, inp, stride, groups=inp)
            self.bn_rep = builder.batchnorm(inp)

        self.relu = builder.activation()
        self.conv2 = builder.conv1x1(inp, oup)
        self.bn2 = builder.batchnorm(oup)

    def forward(self, x):
        if self.deploy:
            out = self.conv_deploy(x)
        else:
            out_1, mask = self.conv1(x)
            out_1 = self.bn1(out_1)
            out_2 = self.conv_rep(x, mask)
            out_2 = self.bn_rep(out_2)
            out = out_1 + out_2

        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = self.relu(out)

        return out


class MobileNetV1(nn.Module):
    def __init__(self, width_mult=1, deploy=False):
        super(MobileNetV1, self).__init__()
        builder = get_builder()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, padding=1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        self.model = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(builder, 32, 64, 1, deploy),
            conv_dw(builder, 64, 128, 2, deploy),
            conv_dw(builder, 128, 128, 1, deploy),
            conv_dw(builder, 128, 256, 2, deploy),
            conv_dw(builder, 256, 256, 1, deploy),
            conv_dw(builder, 256, 512, 2, deploy),
            conv_dw(builder, 512, 512, 1, deploy),
            conv_dw(builder, 512, 512, 1, deploy),
            conv_dw(builder, 512, 512, 1, deploy),
            conv_dw(builder, 512, 512, 1, deploy),
            conv_dw(builder, 512, 512, 1, deploy),
            conv_dw(builder, 512, 1024, 2, deploy),
            conv_dw(builder, 1024, 1024, 1, deploy),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024,1000)
    

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.flatten(1)
