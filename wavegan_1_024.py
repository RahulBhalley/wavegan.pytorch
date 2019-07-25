# -*- coding: utf-8 -*-

from __future__ import print_function

"""
    Based on 128x128 image version of GAN-QP
    Generates 1.024 seconds of sound.
"""

__author__ = "Rahul Bhalley"

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import *

SOUND_DIM = 128 ** 2

###################
# Neural Networks #
###################
# --------------- #
# Generator       #
# --------------- #
# Critic   #
# --------------- #
###################


#############
# Generator #
#############

class MyConvTranspose1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=11, upsample=None, output_padding=1):
        super(MyConvTranspose1d, self).__init__()
        self.upsample = upsample

        #self.upsample_layer = torch.nn.Upsample(scale_factor=upsample)  # TODO: Use `F.interpolate` instead.
        self.upsample_layer = F.interpolate
        reflection_pad = kernel_size // 2
        self.reflection_pad = nn.ConstantPad1d(reflection_pad, value=0)
        self.conv1d = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.convtrans1d = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, output_padding)

    def forward(self, x):
        if self.upsample:
            return self.conv1d(self.reflection_pad(self.upsample_layer(x, scale_factor=self.upsample)))
        else:
            return self.convtrans1d(x)

class Generator(nn.Module):
    """
    Down scaling by a factor of 4
        combination #1
        k = 16 | s = 4 | p = 6
        combination #2 (I don't like this one)
        k = 25 | s = 4 | p = 11 | out_p = 1
        combination #3
        k = 36 | s = 4 | p = 16
        combination #4
        k = 50 | s = 4 | p = ? | out_p = ?
    """

    def __init__(self, model_size=64, kernel_size=25, stride=4, padding=11, upsample=True, output_padding=1, post_proc_filt_len=512):
        super(Generator, self).__init__()

        self.model_size = model_size
        self.post_proc_filt_len = post_proc_filt_len

        if upsample:
            stride = 1
            self.stride = 1
            upsample = 4

        block1 = [
            nn.Linear(Z_DIM, 4 * 4 * model_size * 16),
            #nn.BatchNorm1d(4 * 4 * model_size * 16),
            nn.ReLU()
        ]

        block2 = [
            MyConvTranspose1d(model_size * 16, model_size * 8, kernel_size, stride, padding, upsample, output_padding),
            #nn.BatchNorm1d(model_size * 8),
            nn.ReLU()
        ]

        block3 = [
            MyConvTranspose1d(model_size * 8, model_size * 4,  kernel_size, stride, padding, upsample, output_padding),
            #nn.BatchNorm1d(model_size * 4),
            nn.ReLU()
        ]

        block4 = [
            MyConvTranspose1d(model_size * 4, model_size * 2,  kernel_size, stride, padding, upsample, output_padding),
            #nn.BatchNorm1d(model_size * 2),
            nn.ReLU()
        ]

        block5 = [
            MyConvTranspose1d(model_size * 2, model_size,      kernel_size, stride, padding, upsample, output_padding),
            #nn.BatchNorm1d(model_size),
            nn.ReLU()
        ]

        block6 = [
            MyConvTranspose1d(model_size,     N_CHANNELS,      kernel_size, stride, padding, upsample, output_padding),
            nn.Tanh()
        ]

        if post_proc_filt_len:
            self.ppfilter = nn.Conv1d(N_CHANNELS, N_CHANNELS, post_proc_filt_len)

        all_blocks = block2 + block3 + block4 + block5 + block6
        self.main1 = nn.Sequential(*block1)
        self.main2 = nn.Sequential(*all_blocks)

        # Initalization with uniform Xavier Glorot technique
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        # Free some memory
        del all_blocks, block1, block2, block3, block4, block5, block6

    def forward(self, x):
        for layer in self.main1:
            x = layer(x)
        x = x.view(-1, self.model_size * 16, 16)
        for layer in self.main2:
            x = layer(x)
        if self.post_proc_filt_len:
            # Pad for "same" filtering
            if self.post_proc_filt_len % 2 == 0:
                pad_left = self.post_proc_filt_len // 2
                pad_right = pad_left - 1
            else:
                pad_left = (self.post_proc_filt_len - 1) // 2
                pad_right = pad_left
            x = self.ppfilter(F.pad(x, (pad_left, pad_right)))
        return x
    
    def summary(self):
        x = torch.zeros(BATCH_SIZE, Z_DIM)
        
        # Print the title in a good design
        # for easy recognition.
        print()
        summary_title = '| {} Summary |'.format(self.__class__.__name__)
        for _ in range(len(summary_title)):
            print('-', end='')
        print()
        print(summary_title)
        for _ in range(len(summary_title)):
            print('-', end='')
        print('\n')
        
        # Run forward pass while not tracking history on
        # tape using `torch.no_grad()` for printing the
        # output shape of each neural layer operation.
        print('Input: {}'.format(x.size()))
        with torch.no_grad():
            for layer in self.main1:
                x = layer(x)
                print('Out: {} \tLayer: {}'.format(x.size(), layer))
            x = x.view(-1, self.model_size * 16, 16)  # Reshape for convolution
            print('Out: {} \tLayer: {}'.format(x.size(), 'Reshape'))
            for layer in self.main2:
                x = layer(x)
                print('Out: {} \tLayer: {}'.format(x.size(), layer))


#################
# Critic #
#################

class PhaseShuffle(nn.Module):
    """
    Performs phase shuffling, i.e. shifting feature axis of a 3D tensor
    by a random integer in {-n, n} and performing reflection padding where
    necessary.
    """
    # Adapted from https://github.com/jtcramer/wavegan/
    def __init__(self, shift_factor):
        super(PhaseShuffle, self).__init__()
        self.shift_factor = shift_factor

    def forward(self, x):
        if self.shift_factor == 0:
            return x
        # uniform in (L, R)
        k_list = torch.Tensor(x.shape[0]).random_(0, 2 * self.shift_factor + 1) - self.shift_factor
        k_list = k_list.numpy().astype(int)

        # Combine sample indices into lists so that less shuffle operations
        # need to be performed
        k_map = {}
        for idx, k in enumerate(k_list):
            k = int(k)
            if k not in k_map:
                k_map[k] = []
            k_map[k].append(idx)

        # Make a copy of x for our output
        x_shuffle = x.clone()

        # Apply shuffle to each sample
        for k, idxs in k_map.items():
            if k > 0:
                x_shuffle[idxs] = F.pad(x[idxs][..., :-k], (k, 0), mode='reflect')
            else:
                x_shuffle[idxs] = F.pad(x[idxs][..., -k:], (0, -k), mode='reflect')

        assert x_shuffle.shape == x.shape, "{}, {}".format(x_shuffle.shape, x.shape)
        return x_shuffle

class Critic(nn.Module):
    """
    Up scaling by a factor of 4
        combination #1
        k = 16 | s = 4 | p = 6
        combination #2 (I don't like this one)
        k = 25 | s = 4 | p = 11
        combination #3
        k = 36 | s = 4 | p = 16
    """

    def __init__(self, model_size=64, kernel_size=25, stride=4, padding=11, leak=0.2, shift_factor=2):
        super(Critic, self).__init__()

        block1 = [
            nn.Conv1d(N_CHANNELS,     model_size,      kernel_size, stride, padding, bias=False),
            nn.LeakyReLU(leak),
            PhaseShuffle(shift_factor)
        ]

        block2 = [
            nn.Conv1d(model_size,     model_size * 2,  kernel_size, stride, padding, bias=False),
            nn.BatchNorm1d(model_size * 2),
            nn.LeakyReLU(leak),
            PhaseShuffle(shift_factor)
        ]
        
        block3 = [
            nn.Conv1d(model_size * 2, model_size * 4,  kernel_size, stride, padding, bias=False),
            nn.BatchNorm1d(model_size * 4),
            nn.LeakyReLU(leak),
            PhaseShuffle(shift_factor)
        ]

        block4 = [
            nn.Conv1d(model_size * 4, model_size * 8,  kernel_size, stride, padding, bias=False),
            nn.BatchNorm1d(model_size * 8),
            nn.LeakyReLU(leak),
            PhaseShuffle(shift_factor)
        ]

        block5 = [
            nn.Conv1d(model_size * 8, model_size * 16, kernel_size, stride, padding, bias=False),
            nn.BatchNorm1d(model_size * 16),
            nn.LeakyReLU(leak)
        ]

        all_blocks = block1 + block2 + block3 + block4 + block5
        self.conv = nn.Sequential(*all_blocks)
        self.linear = nn.Linear(4 * 4 * model_size * 16, 1, bias=False)

        # Initalization with uniform Xavier Glorot technique
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        # Free some memory
        del all_blocks, block1, block2, block3, block4, block5

    def forward(self, x):
        for layer in self.conv:
            x = layer(x)
        x = x.view(-1, self.num_flat_features(x))  # flatten the conv output
        x - self.linear(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def summary(self):
        x = torch.zeros(BATCH_SIZE, N_CHANNELS, SOUND_DIM)
        
        # Print the title in a good design
        # for easy recognition.
        print()
        summary_title = '| {} Summary |'.format(self.__class__.__name__)
        for _ in range(len(summary_title)):
            print('-', end='')
        print()
        print(summary_title)
        for _ in range(len(summary_title)):
            print('-', end='')
        print('\n')
        
        # Run forward pass while not tracking history on
        # tape using `torch.no_grad()` for printing the
        # output shape of each neural layer operation.
        print('Input: {}'.format(x.size()))
        with torch.no_grad():
            for layer in self.conv:
                x = layer(x)
                print('Out: {} \tLayer: {}'.format(x.size(), layer))
            x = x.view(-1, self.num_flat_features(x))  # flatten the conv output
            print('Out: {} \tlayer: {}'.format(x.size(), 'Reshape: Flatten'))
            x = self.linear(x)
            print('Out: {} \tLayer: {}'.format(x.size(), layer))


if __name__ == "__main__":

    g_model = Generator()
    g_model.summary()

    c_model = Critic()
    c_model.summary()