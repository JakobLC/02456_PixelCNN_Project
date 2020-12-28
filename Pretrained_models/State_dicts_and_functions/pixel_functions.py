# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 13:35:32 2020

@author: lowes
"""
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim, cuda, backends
from torch.autograd import Variable
from torch.utils import data
from torchvision import datasets, transforms, utils

class DynamicBinarization():
    def __call__(self,x):
        return x.bernoulli()*2-1
def Binarize(x):
    return   x.bernoulli()*2-1

class MaskedConv2d(nn.Conv2d):
    
    def __init__(self, *args, mask_type='B', **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        
        # Mask A) without center pixel
        # Mask B) with center pixel

        # 1 1 1 1 1
        # 1 1 1 1 1
        # 1 1 X 0 0
        # 0 0 0 0 0
        # 0 0 0 0 0

        self.mask = torch.ones_like(self.weight)
        _, _, height, width = self.weight.size()
        
        self.mask[:, :, height // 2, width // 2 + (1 if mask_type=='B' else 0):] = 0
        self.mask[:, :, height // 2 + 1:] = 0

        if cuda:
            self.mask = self.mask.cuda()
        
    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)

class CroppedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(CroppedConv2d, self).__init__(*args, **kwargs)

    def forward(self, x):
        x = super(CroppedConv2d, self).forward(x)

        kernel_height, _ = self.kernel_size
        res = x[:, :, 1:-kernel_height, :]
        shifted_up_res = x[:, :, :-kernel_height-1, :]

        return res, shifted_up_res

class CausalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(CausalBlock, self).__init__()
        self.split_size = out_channels

        self.v_conv = CroppedConv2d(in_channels,
                                    2 * out_channels,
                                    (kernel_size // 2 + 1, kernel_size),
                                    padding=(kernel_size // 2 + 1, kernel_size // 2))
        self.v_fc = nn.Conv2d(in_channels,
                              2 * out_channels,
                              (1, 1))
        self.v_to_h = nn.Conv2d(2 * out_channels,
                                2 * out_channels,
                                (1, 1))

        self.h_conv = MaskedConv2d(in_channels,
                                   2 * out_channels,
                                   (1, kernel_size),
                                   padding=(0, kernel_size // 2),
                                   mask_type='A')
        self.h_fc = nn.Conv2d(out_channels,
                              out_channels,
                              (1, 1))

    def forward(self, image):
        v_out, v_shifted = self.v_conv(image)
        v_out += self.v_fc(image)
        v_out_tanh, v_out_sigmoid = torch.split(v_out, self.split_size, dim=1)
        v_out = torch.tanh(v_out_tanh) * torch.sigmoid(v_out_sigmoid)

        h_out = self.h_conv(image)
        v_shifted = self.v_to_h(v_shifted)
        h_out += v_shifted
        h_out_tanh, h_out_sigmoid = torch.split(h_out, self.split_size, dim=1)
        h_out = torch.tanh(h_out_tanh) * torch.sigmoid(h_out_sigmoid)
        h_out = self.h_fc(h_out)

        return v_out, h_out


class GatedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(GatedBlock, self).__init__()
        self.split_size = out_channels

        self.v_conv = CroppedConv2d(in_channels,
                                    2 * out_channels,
                                    (kernel_size // 2 + 1, kernel_size),
                                    padding=(kernel_size // 2 + 1, kernel_size // 2))
        self.v_fc = nn.Conv2d(in_channels,
                              2 * out_channels,
                              (1, 1))
        self.v_to_h = nn.Conv2d(2 * out_channels,
                                2 * out_channels,
                                (1, 1))

        self.h_conv = MaskedConv2d(in_channels,
                                   2 * out_channels,
                                   (1, kernel_size),
                                   padding=(0, kernel_size // 2),
                                   mask_type='B')
        self.h_fc = nn.Conv2d(out_channels,
                              out_channels,
                              (1, 1))

        self.h_skip = nn.Conv2d(out_channels,
                                out_channels,
                                (1, 1))

        self.label_embedding = nn.Embedding(10, 2*out_channels)

    def forward(self, x):
        v_in, h_in, skip, label = x[0], x[1], x[2], x[3]

        label_embedded = self.label_embedding(label).unsqueeze(2).unsqueeze(3)

        v_out, v_shifted = self.v_conv(v_in)
        v_out += self.v_fc(v_in)
        v_out += label_embedded
        v_out_tanh, v_out_sigmoid = torch.split(v_out, self.split_size, dim=1)
        v_out = torch.tanh(v_out_tanh) * torch.sigmoid(v_out_sigmoid)

        h_out = self.h_conv(h_in)
        v_shifted = self.v_to_h(v_shifted)
        h_out += v_shifted
        h_out += label_embedded
        h_out_tanh, h_out_sigmoid = torch.split(h_out, self.split_size, dim=1)
        h_out = torch.tanh(h_out_tanh) * torch.sigmoid(h_out_sigmoid)

        # skip connection
        skip = skip + self.h_skip(h_out)

        h_out = self.h_fc(h_out)

        # residual connections
        h_out = h_out + h_in
        v_out = v_out + v_in

        return {0: v_out, 1: h_out, 2: skip, 3: label}


class GatedPixelCNN(nn.Module):
    def __init__(self, hidden_fmaps=64, causal_ksize=7, hidden_ksize=3, num_layers=12, out_hidden_fmaps=256, color_levels=1):
        super(GatedPixelCNN, self).__init__()

        DATA_CHANNELS = 1

        self.color_levels = color_levels
        self.hidden_fmaps = hidden_fmaps

        self.causal_conv = CausalBlock(DATA_CHANNELS,
                                       hidden_fmaps,
                                       causal_ksize)

        self.hidden_conv = nn.Sequential(
            *[GatedBlock(hidden_fmaps, hidden_fmaps, hidden_ksize) for _ in range(num_layers)]
        )

        self.label_embedding = nn.Embedding(10, self.hidden_fmaps)

        self.out_hidden_conv = nn.Conv2d(hidden_fmaps,
                                         out_hidden_fmaps,
                                         (1, 1))

        self.out_conv = nn.Conv2d(out_hidden_fmaps,
                                  color_levels,
                                  (1, 1))

    def forward(self, image, label):
        count, data_channels, height, width = image.size()

        v, h = self.causal_conv(image)

        _, _, out, _ = self.hidden_conv({0: v,
                                         1: h,
                                         2: image.new_zeros((count, self.hidden_fmaps, height, width), requires_grad=True),
                                         3: label}).values()

        label_embedded = self.label_embedding(label).unsqueeze(2).unsqueeze(3)

        # add label bias
        out += label_embedded
        out = F.relu(out)
        out = F.relu(self.out_hidden_conv(out))
        out = self.out_conv(out)

        out = out.view(count, self.color_levels, height, width)

        return out
         

class ResNetBlock(nn.Module):
    
    def __init__(self, num_filters=128):
        super(ResNetBlock, self).__init__()
        
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters//2, kernel_size=1),
            nn.ReLU(),
            MaskedConv2d(in_channels=num_filters//2, out_channels=num_filters//2, kernel_size=3, padding=1, mask_type='B'),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_filters//2, out_channels=num_filters, kernel_size=1)
        )
        
    def forward(self, x):
        return self.layers(x) + x

class PixelCNN(nn.Module):
    def __init__(self, num_layers=12, num_filters=128,color_levels=1):
        super(PixelCNN, self).__init__()
        
        layers = [MaskedConv2d(in_channels=1,
                               out_channels=num_filters,
                               kernel_size=7,
                               padding=3, mask_type='A')]
        
        for _ in range(num_layers):
            layers.append(ResNetBlock(num_filters=num_filters))
            
        layers.extend([
            nn.ReLU(),
            nn.Conv2d(in_channels=num_filters, out_channels=256, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=color_levels, kernel_size=1)
        ])
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class GatedBlock_space(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(GatedBlock_space, self).__init__()
        self.split_size = out_channels

        self.v_conv = CroppedConv2d(in_channels,
                                    2 * out_channels,
                                    (kernel_size // 2 + 1, kernel_size),
                                    padding=(kernel_size // 2 + 1, kernel_size // 2))
        self.v_fc = nn.Conv2d(in_channels,
                              2 * out_channels,
                              (1, 1))
        self.v_to_h = nn.Conv2d(2 * out_channels,
                                2 * out_channels,
                                (1, 1))

        self.h_conv = MaskedConv2d(in_channels,
                                   2 * out_channels,
                                   (1, kernel_size),
                                   padding=(0, kernel_size // 2),
                                   mask_type='B')
        self.h_fc = nn.Conv2d(out_channels,
                              out_channels,
                              (1, 1))

        self.h_skip = nn.Conv2d(out_channels,
                                out_channels,
                                (1, 1))

        self.label_embedding = nn.Embedding(10, 2*out_channels)
        self.X_embedding = nn.Embedding(10, 28)
        self.Y_embedding = nn.Embedding(10, 28)

    def forward(self, x):
        v_in, h_in, skip, label = x[0], x[1], x[2], x[3]

        label_embedded = self.label_embedding(label).unsqueeze(2).unsqueeze(3)
        X_embedded = self.X_embedding(label).unsqueeze(1).unsqueeze(3)
        Y_embedded = self.Y_embedding(label).unsqueeze(1).unsqueeze(2)

        v_out, v_shifted = self.v_conv(v_in)
        v_out += self.v_fc(v_in)
        v_out += label_embedded
        v_out_tanh, v_out_sigmoid = torch.split(v_out, self.split_size, dim=1)
        v_out = torch.tanh(v_out_tanh) * torch.sigmoid(v_out_sigmoid)

        h_out = self.h_conv(h_in)
        v_shifted = self.v_to_h(v_shifted)
        h_out += v_shifted
        h_out += label_embedded
        h_out += X_embedded
        h_out += Y_embedded
        h_out_tanh, h_out_sigmoid = torch.split(h_out, self.split_size, dim=1)
        h_out = torch.tanh(h_out_tanh) * torch.sigmoid(h_out_sigmoid)

        # skip connection
        skip = skip + self.h_skip(h_out)

        h_out = self.h_fc(h_out)

        # residual connections
        h_out = h_out + h_in
        v_out = v_out + v_in

        return {0: v_out, 1: h_out, 2: skip, 3: label}


class GatedPixelCNN_space(nn.Module):
    def __init__(self, hidden_fmaps=64, causal_ksize=7, hidden_ksize=3, num_layers=12, out_hidden_fmaps=256, color_levels=1):
        super(GatedPixelCNN_space, self).__init__()

        DATA_CHANNELS = 1

        self.color_levels = color_levels
        self.hidden_fmaps = hidden_fmaps

        self.causal_conv = CausalBlock(DATA_CHANNELS,
                                       hidden_fmaps,
                                       causal_ksize)

        self.hidden_conv = nn.Sequential(
            *[GatedBlock_space(hidden_fmaps, hidden_fmaps, hidden_ksize) for _ in range(num_layers)]
        )

        self.label_embedding = nn.Embedding(10, self.hidden_fmaps)
        self.X_embedding = nn.Embedding(10, 28)
        self.Y_embedding = nn.Embedding(10, 28)

        self.out_hidden_conv = nn.Conv2d(hidden_fmaps,
                                         out_hidden_fmaps,
                                         (1, 1))

        self.out_conv = nn.Conv2d(out_hidden_fmaps,
                                  color_levels,
                                  (1, 1))

    def forward(self, image, label):
        count, data_channels, height, width = image.size()

        v, h = self.causal_conv(image)

        _, _, out, _ = self.hidden_conv({0: v,
                                         1: h,
                                         2: image.new_zeros((count, self.hidden_fmaps, height, width), requires_grad=True),
                                         3: label}).values()

        label_embedded = self.label_embedding(label).unsqueeze(2).unsqueeze(3)
        X_embedded = self.X_embedding(label).unsqueeze(1).unsqueeze(3)
        Y_embedded = self.Y_embedding(label).unsqueeze(1).unsqueeze(2)

        # add label bias
        out += label_embedded
        out += X_embedded
        out += Y_embedded
        out = F.relu(out)
        out = F.relu(self.out_hidden_conv(out))
        out = self.out_conv(out)

        out = out.view(count, self.color_levels, height, width)

        return out

def sample_images(net,num_colors=1,num_samples=8,label_bool=True):
    sample = torch.Tensor(num_samples**2, 1, 28, 28).cuda()
    sample.fill_(0)
    if label_bool:
        label = torch.randint(high=10, size=(num_samples**2,)).long().cuda()

    with torch.no_grad():
        for i in range(28):
            for j in range(28):
                if num_colors == 1:
                    if label_bool:
                        out = net(Variable(Binarize(sample)),label)
                    else:
                        out = net(Variable(Binarize(sample)))
                    probs = torch.sigmoid(out[:, :, i, j]).data
                    sample[:, :, i, j] = torch.bernoulli(probs).cuda()
                else:
                    if label_bool:
                        out = net(sample*2-1,label)
                    else:
                        out = net(sample*2-1)
                    probs = torch.softmax(out[:, :, i, j],1).data
                    sample[:, :, i, j] = torch.multinomial(probs, 1).float()/(num_colors-1)
    return sample   

def plot_half_boys(net,input,num_colors, label=None):
  # Test of generating images from half an image
    with torch.no_grad():
        if num_colors==1:
            target = Variable(input.data[:,0].unsqueeze(1))/2+1/2*torch.ones_like(input)
        else:
            target = Variable(input.data[:,0].unsqueeze(1))
        #sample = torch.zeros_like(target).cuda()
        #sample.fill_(0)
        sample = torch.Tensor(100, 1, 28, 28).cuda()
        sample.fill_(0)
        sample[:,:,0:14,:]=target[:,:,0:14,:]
        for i in range(14,28):
            for j in range(28):
                if num_colors == 1:
                    if label is None:
                        out = net(Variable(Binarize(sample)))
                    else:
                        out = net(Variable(Binarize(sample)),label)
                    probs = torch.sigmoid(out[:, :, i, j]).data
                    sample[:, :, i, j] = torch.bernoulli(probs).cuda()
                else:
                    if label is None:
                        out = net(sample*2-1)
                    else:
                        out = net(sample*2-1,label)
                    probs = torch.softmax(out[:, :, i, j],1).data
                    sample[:, :, i, j] = torch.multinomial(probs, 1).float()/(num_colors-1)
                    
        return sample

    