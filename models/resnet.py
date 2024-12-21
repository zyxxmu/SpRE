import torch.nn as nn

from utils.builder import get_builder
from args import args

# BasicBlock {{{
class BasicBlock(nn.Module):
    M = 2
    expansion = 1

    def __init__(self, builder, inplanes, planes, stride=1, downsample=None, deploy=False):
        super(BasicBlock, self).__init__()
        self.deploy = deploy 
        if self.deploy:
            self.conv1_deploy = builder.conv3x3(inplanes, planes, stride, bias=True)
            self.conv2_deploy = builder.conv3x3(inplanes, planes, bias=True)
        else:
            self.conv1 = builder.conv_rep_conv(3, inplanes, planes, stride)
            self.bn1 = builder.batchnorm(planes)
            self.conv_rep1 = builder.rep_conv(3, inplanes, planes, stride)
            self.bn_rep1 = builder.batchnorm(planes)

            self.conv2 = builder.conv_rep_conv(3, planes, planes)
            self.bn2 = builder.batchnorm(planes, last_bn=True)
            self.conv_rep2 = builder.rep_conv(3, planes, planes)
            self.bn_rep2 = builder.batchnorm(planes, last_bn=True)
        
        self.relu = builder.activation()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        if self.deploy:
            out = self.conv1(x)
            out = self.relu(out)
            out = self.conv2(out)
        else:
            out_a, mask = self.conv1(x)
            out_a = self.bn1(out_a)    
            out_b = self.conv_rep1(x, mask)
            out_b = self.bn_rep1(out_b)
            out = out_a + out_b
            out = self.relu(out)

            out_a, mask = self.conv2(out)
            out_a = self.bn2(out_a)    
            out_b = self.conv_rep2(out, mask)
            out_b = self.bn_rep2(out_b)
            out = out_a + out_b

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# Bottleneck {{{
class Bottleneck(nn.Module):
    M = 3
    expansion = 4

    def __init__(self, builder, inplanes, planes, stride=1, downsample=None, deploy=False):
        super(Bottleneck, self).__init__()
        self.deploy = deploy
        self.conv1 = builder.conv1x1(inplanes, planes)
        self.bn1 = builder.batchnorm(planes)
        
        if self.deploy == True:
            self.conv2_deploy = builder.conv3x3(planes, planes, stride, bias=True)
        else:
            self.conv2 = builder.conv_rep_conv(3, planes, planes, stride=stride)
            self.bn2 = builder.batchnorm(planes)
            self.conv_rep = builder.rep_conv(3, planes, planes, stride=stride)
            self.bn_rep = builder.batchnorm(planes)

        self.conv3 = builder.conv1x1(planes, planes * self.expansion)
        self.bn3 = builder.batchnorm(planes * self.expansion, last_bn=True)
        self.relu = builder.activation()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        if self.deploy:
            out = self.conv2_deploy(out)
        else:
            out_1, mask = self.conv2(out)
            out_1 = self.bn2(out_1)
            out_2 = self.conv_rep(out, mask)
            out_2 = self.bn_rep(out_2)
            out = out_1 + out_2

        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, builder, block, layers, num_classes=1000, deploy=False):
        self.inplanes = 64
        self.deploy = deploy
        super(ResNet, self).__init__()
        
        width_1 = 64
        width_2 = 128
        width_3 = 256
        width_4 = 512

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=True)
        self.bn1 = builder.batchnorm(width_1)
        self.relu = builder.activation()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(builder, block, width_1, layers[0])
        self.layer2 = self._make_layer(builder, block, width_2, layers[1], stride=2)
        self.layer3 = self._make_layer(builder, block, width_3, layers[2], stride=2)
        self.layer4 = self._make_layer(builder, block, width_4, layers[3], stride=2)
        self.avgpool = nn.Sequential(nn.AvgPool2d(7))
        #self.fc = nn.Linear(width_4 * block.expansion, num_classes)
        self.fc = builder.conv1x1_fc(width_4 * block.expansion, num_classes)
    

    def _make_layer(self, builder, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            dconv = builder.conv1x1(
                self.inplanes, planes * block.expansion, stride=stride
            )
            dbn = builder.batchnorm(planes * block.expansion)
            if dbn is not None:
                downsample = nn.Sequential(dconv, dbn)
            else:
                downsample = dconv

        layers = []
        layers.append(block(builder, self.inplanes, planes, stride, downsample, deploy=self.deploy))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(builder, self.inplanes, planes, deploy=self.deploy))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        if self.bn1 is not None:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
            
        #x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(x.size(0), -1)
        #import pdb; pdb.set_trace()
        return x


def ResNet18(deploy):
    return ResNet(get_builder(), BasicBlock, [2, 2, 2, 2], num_classes=1000, deploy=deploy)


def ResNet50(deploy):
    return ResNet(get_builder(), Bottleneck, [3, 4, 6, 3], num_classes=1000, deploy=deploy)
