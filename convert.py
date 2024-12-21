import os
import pathlib
import random
import shutil
import time
import datetime
import json

import torch
import torch.nn as nn
from models.resnet import Bottleneck, BasicBlock
from models.resnet_cifar import ResBasicBlock
from models.mobilenetv1 import conv_dw
from args import args
import torch.backends.cudnn as cudnn
import models
from trainer import train, validate

import math
import data

from utils.logging import *

def set_gpu(args, model):
    # DataParallel will divide and allocate batch_size to all available GPUs
    print(f"=> Parallelizing on {args.gpus} gpus")
    torch.cuda.set_device(args.gpus[0])
    args.gpu = args.gpus[0]
    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda(
        args.gpus[0]
    )
    cudnn.benchmark = True

    return model

def fuse_bn_tensor(conv, bn):
    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std

def get_equivalent_kernel_bias(conv1, bn1, conv2, bn2):
    w = conv1.get_sparse_weights()
    conv1.weight.data = w
    b = torch.sign(torch.abs(w))
    sparse_weight = conv2.weight * b
    sparse_weight = sparse_weight * conv2.mask
    conv2.weight.data = sparse_weight
    conv1,bias1 = fuse_bn_tensor(conv1, bn1)
    conv2,bias2 = fuse_bn_tensor(conv2, bn2)
    return conv1+conv2, bias1+bias2

def get_dataset(args):
    print(f"=> Getting {args.set} dataset")
    dataset = getattr(data, args.set)(args)
    return dataset

criterion = nn.CrossEntropyLoss().cuda()
logger = get_logger(os.path.join('logger.log'))
data = get_dataset(args)

model = models.__dict__[args.arch](deploy=False)
checkpoint = torch.load(args.train_model_link)
checkpoint = checkpoint['state_dict']
ckpt = {k.replace('module.', ''): v for k, v in checkpoint.items()}
model.load_state_dict(ckpt) 

model_deploy = models.__dict__[args.arch](deploy=True)

for n, m in model.named_modules():
    if args.arch == 'ResNet50':
        if isinstance(m, Bottleneck):
            conv_deploy = get_equivalent_kernel_bias(m.conv2, m.bn2, m.conv_rep, m.bn_rep)
            ckpt[n+'.conv2_deploy.weight']=conv_deploy[0]
            ckpt[n+'.conv2_deploy.bias']=conv_deploy[1]
    elif args.arch == 'ResNet18' or args.arch == 'ResNet32':
        if isinstance(m, BasicBlock) or isinstance(m, ResBasicBlock):
            conv_deploy = get_equivalent_kernel_bias(m.conv1, m.bn1, m.conv_rep1, m.bn_rep1)
            ckpt[n+'.conv1_deploy.weight']=conv_deploy[0]
            ckpt[n+'.conv1_deploy.bias']=conv_deploy[1]
            conv_deploy = get_equivalent_kernel_bias(m.conv2, m.bn2, m.conv_rep2, m.bn_rep2)
            ckpt[n+'.conv2_deploy.weight']=conv_deploy[0]
            ckpt[n+'.conv2_deploy.bias']=conv_deploy[1]
    else:
        if isinstance(m, conv_dw):
            conv_deploy = get_equivalent_kernel_bias(m.conv1, m.bn1, m.conv_rep, m.bn_rep)
            ckpt[n+'.conv_deploy.weight']=conv_deploy[0]
            ckpt[n+'.conv_deploy.bias']=conv_deploy[1]

torch.save(ckpt, args.arch+'_deploy'+str(args.N)+str(args.M)+'.pth')
model_deploy.load_state_dict(ckpt, strict=False)

print("Before Re-parameterization:")
set_gpu(args, model)
acc1, acc5 = validate(
            data.val_loader, model, criterion, args, logger, epoch=1
        )

print("After Re-parameterization:")
set_gpu(args, model_deploy)
acc1, acc5 = validate(
            data.val_loader, model_deploy, criterion, args, logger, epoch=1
        )

