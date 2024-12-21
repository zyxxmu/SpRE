from torch.nn import init
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import math

from args import args
import numpy as np

DenseConv = nn.Conv2d
 
class SpReConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros', N=args.N, M=args.M, **kwargs):
        super(SpReConv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, **kwargs)
        self.mask = nn.Parameter(torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]]), requires_grad=False)
        self.N = N
        self.M = M

    def set_rep_masks(self, w):
        weight_tmp = w.detach().view(-1)
        prune_rate = 1 - (self.N / self.M)
        prune_num = int(weight_tmp.size(0) * prune_rate)
        index = torch.argsort(torch.abs(weight_tmp), dim=0)[:prune_num]
        w_b = torch.ones(weight_tmp.shape).to(weight_tmp.device)
        w_b = w_b.scatter_(dim=0, index=index, value=0).reshape(self.weight.shape)
        shape = w_b.shape           
        w_b = w_b.view(int(shape[0]*shape[1]), int(shape[2]*shape[3]))
        sparsity = 1 - (torch.sum(w_b, 0).view(shape[2], shape[3])/int(shape[0]*shape[1]))
        mask = sparsity < prune_rate
        self.mask.data = mask.float().to(self.mask.device)

    def forward(self, x, w):
        b = torch.sign(torch.abs(w))
        sparse_weight = self.weight * b
        sparse_weight = sparse_weight * self.mask
        out = F.conv2d(
            x, sparse_weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )

        return out

class SparseConv(nn.Conv2d):
    """" implement sparse convolution layer """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros', N=args.N, M=args.M, **kwargs):
        self.N = N
        self.M = M
        super(SparseConv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, **kwargs)

    def get_sparse_weights(self):
        return Sparse_NHWC.apply(self.weight, self.N, self.M)

    def get_vanilla_sparse_weights(self):
        weight_tmp = self.weight.detach().cpu().view(-1)
        prune_num = int(weight_tmp.size(0) * 0.9)
        index = torch.argsort(torch.abs(weight_tmp), dim=0)[:prune_num]
        w_b = torch.ones(weight_tmp.shape)
        w_b = w_b.scatter_(dim=0, index=index, value=0).reshape(self.weight.shape)
        return w_b
        
    def forward(self, x):
        w = self.get_sparse_weights()
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

class SparseConv_ASP(nn.Conv2d):
    """" implement sparse convolution layer """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros', N=args.N, M=args.M, **kwargs):
        self.N = N
        self.M = M
        super(SparseConv_ASP, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, **kwargs)
        self.mask = torch.ones(self.weight.shape)

    def set_mask(self):
        length = self.weight.numel()
        group = int(length/self.M)

        weight_temp = self.weight.detach().abs().permute(0,2,3,1).reshape(group, self.M)
        index = torch.argsort(weight_temp, dim=1)[:, :int(self.M-self.N)]

        w_b = torch.ones(weight_temp.shape, device=weight_temp.device)
        w_b = w_b.scatter_(dim=1, index=index, value=0).reshape(self.weight.permute(0,2,3,1).shape)
        w_b = w_b.permute(0,3,1,2)

        self.mask = w_b.to(self.weight.device)
        
        
    def forward(self, x):
        w = self.weight * self.mask.to(self.weight.device)
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x
    
       
class SparseRepConv(nn.Conv2d):
    """" implement sparse convolution layer """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros', N=args.N, M=args.M, **kwargs):
        self.N = N
        self.M = M
        super(SparseRepConv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, **kwargs)

    def get_sparse_weights(self):
        return Sparse_NHWC.apply(self.weight, self.N, self.M)

    def get_vanilla_sparse_weights(self):
        weight_tmp = self.weight.detach().cpu().view(-1)
        prune_num = int(weight_tmp.size(0) * (1 - (self.N / self.M)))
        index = torch.argsort(torch.abs(weight_tmp), dim=0)[:prune_num]
        w_b = torch.ones(weight_tmp.shape)
        w_b = w_b.scatter_(dim=0, index=index, value=0).reshape(self.weight.shape)
        return w_b
        
    def forward(self, x):
        w = self.get_sparse_weights()
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x,w


class SparseRepConv_ASP(nn.Conv2d):
    """" implement sparse convolution layer """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros', N=args.N, M=args.M, **kwargs):
        self.N = N
        self.M = M
        super(SparseRepConv_ASP, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, **kwargs)
        self.mask = torch.ones(self.weight.shape)

    def set_mask(self):
        length = self.weight.numel()
        group = int(length/self.M)

        weight_temp = self.weight.detach().abs().permute(0,2,3,1).reshape(group, self.M)
        index = torch.argsort(weight_temp, dim=1)[:, :int(self.M-self.N)]

        w_b = torch.ones(weight_temp.shape, device=weight_temp.device)
        w_b = w_b.scatter_(dim=1, index=index, value=0).reshape(self.weight.permute(0,2,3,1).shape)
        w_b = w_b.permute(0,3,1,2)

        self.mask = w_b.to(self.weight.device)
    
    def forward(self, x):
        w = self.weight * self.mask.to(self.weight.device)
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x,w


class SparseLinear(nn.Linear):

    def __init__(self, in_features: int, out_features: int, bias: bool = True, N=1, M=16, decay = 0.0002, **kwargs):
        self.N = N
        self.M = M
        super(SparseLinear, self).__init__(in_features, out_features, bias = True)


    def get_sparse_weights(self):
        return Sparse.apply(self.weight, self.N, self.M)

        
    def forward(self, x):

        w = self.get_sparse_weights()
        x = F.linear(x, w, self.bias)
        return x

class Sparse_NHWC(autograd.Function):
    @staticmethod
    def forward(ctx, weight, N, M, decay = 0.0002):
        ctx.save_for_backward(weight)
        output = weight.clone()
        length = weight.numel()
        group = int(length/M)

        weight_temp = weight.detach().abs().permute(0,2,3,1).reshape(group, M)
        index = torch.argsort(weight_temp, dim=1)[:, :int(M-N)]

        w_b = torch.ones(weight_temp.shape, device=weight_temp.device)
        w_b = w_b.scatter_(dim=1, index=index, value=0).reshape(weight.permute(0,2,3,1).shape)
        w_b = w_b.permute(0,3,1,2)

        ctx.mask = w_b
        ctx.decay = decay

        return output*w_b

    @staticmethod
    def backward(ctx, grad_output):
        weight, = ctx.saved_tensors
        return grad_output + ctx.decay * (1-ctx.mask) * weight, None, None

class Sparse(autograd.Function):

    """" Prune the unimprotant weight for the forwards phase but pass the gradient to dense weight using SR-STE in the backwards phase"""
    @staticmethod
    def forward(ctx, weight, N, M, decay = 0.0002):
        ctx.save_for_backward(weight)

        output = weight.clone()
        length = weight.numel()
        group = int(length/M)

        weight_temp = weight.detach().abs().reshape(group, M)
        index = torch.argsort(weight_temp, dim=1)[:, :int(M-N)]

        w_b = torch.ones(weight_temp.shape, device=weight_temp.device)
        w_b = w_b.scatter_(dim=1, index=index, value=0).reshape(weight.shape)
        ctx.mask = w_b
        ctx.decay = decay

        return output*w_b


    @staticmethod
    def backward(ctx, grad_output):

        weight, = ctx.saved_tensors
        return grad_output + ctx.decay * (1-ctx.mask) * weight, None, None





