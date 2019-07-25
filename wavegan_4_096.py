# -*- coding: utf-8 -*-

from __future__ import print_function

"""
    Based on 256x256 image version of GAN-QP
    Generates 4.096 seconds of sound.
"""

__author__ = "Rahul Bhalley"

import torch
import torch.nn as nn

from config import *

SOUND_DIM = 256 ** 2

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

class Generator(nn.Module):
    """
    Down scaling by a factor of 4
        combination #1
        k = 16 | s = 4 | p = 6
        combination #2 (I don't like this one)
        k = 25 | s = 4 | p = 11 | out_p = 1
        combination #3
        k = 36 | s = 4 | p = 16
    """

    def __init__(self, model_size=64, kernel_size=36, stride=4, padding=16):
        super(Generator, self).__init__()

        self.model_size = model_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        block1 = [
            nn.Linear(Z_DIM, 4 * 4 * model_size * 32),
            nn.BatchNorm1d(4 * 4 * model_size * 32),
            nn.ReLU()
        ]

        block2 = [
            nn.ConvTranspose1d(model_size * 32, model_size * 16, kernel_size, stride, padding),
            nn.BatchNorm1d(model_size * 16),
            nn.ReLU()
        ]

        block3 = [
            nn.ConvTranspose1d(model_size * 16, model_size * 8, kernel_size, stride, padding),
            nn.BatchNorm1d(model_size * 8),
            nn.ReLU()
        ]

        block4 = [
            nn.ConvTranspose1d(model_size * 8, model_size * 4,  kernel_size, stride, padding),
            nn.BatchNorm1d(model_size * 4),
            nn.ReLU()
        ]

        block5 = [
            nn.ConvTranspose1d(model_size * 4, model_size * 2,  kernel_size, stride, padding),
            nn.BatchNorm1d(model_size * 2),
            nn.ReLU()
        ]

        block6 = [
            nn.ConvTranspose1d(model_size * 2, model_size,      kernel_size, stride, padding),
            nn.BatchNorm1d(model_size),
            nn.ReLU()
        ]

        block7 = [
            nn.ConvTranspose1d(model_size,     N_CHANNELS,      kernel_size, stride, padding),
            nn.Tanh()
        ]

        all_blocks = block2 + block3 + block4 + block5 + block6 + block7
        self.main1 = nn.Sequential(*block1)
        self.main2 = nn.Sequential(*all_blocks)

        # Initalization with uniform Xavier Glorot technique
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        # Free some memory
        del all_blocks, block1, block2, block3, block4, block5, block6, block7

        # Print summary
        if VERBOSE:
            self.summary()

    def forward(self, x):
        for layer in self.main1:
            x = layer(x)
        x = x.view(-1, self.model_size * 32, 16)
        for layer in self.main2:
            x = layer(x)
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
            x = x.view(-1, self.model_size * 32, 16)  # Reshape for convolution
            print('Out: {} \tLayer: {}'.format(x.size(), 'Reshape'))
            for layer in self.main2:
                x = layer(x)
                print('Out: {} \tLayer: {}'.format(x.size(), layer))


#################
# Critic #
#################

class Critic(nn.Module):
    """
    Up scaling by a factor of 4
        combination #1
        k = 16 | s = 4 | p = 6
        combination #2 (I don't like this one)
        k = 25 | s = 4 | p = 11 | out_p = 1
        combination #3
        k = 36 | s = 4 | p = 16
    """

    def __init__(self, model_size=64, kernel_size=36, stride=4, padding=16, leak=0.2):
        super(Critic, self).__init__()

        self.model_size = model_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.leak = leak

        block1 = [
            nn.Conv1d(N_CHANNELS,     model_size,      kernel_size, stride, padding, bias=False),
            nn.LeakyReLU(self.leak)
        ]

        block2 = [
            nn.Conv1d(model_size,     model_size * 2,  kernel_size, stride, padding, bias=False),
            nn.BatchNorm1d(model_size * 2),
            nn.LeakyReLU(self.leak)
        ]
        
        block3 = [
            nn.Conv1d(model_size * 2, model_size * 4,  kernel_size, stride, padding, bias=False),
            nn.BatchNorm1d(model_size * 4),
            nn.LeakyReLU(self.leak)
        ]

        block4 = [
            nn.Conv1d(model_size * 4, model_size * 8,  kernel_size, stride, padding, bias=False),
            nn.BatchNorm1d(model_size * 8),
            nn.LeakyReLU(self.leak)
        ]

        block5 = [
            nn.Conv1d(model_size * 8, model_size * 16, kernel_size, stride, padding, bias=False),
            nn.BatchNorm1d(model_size * 16),
            nn.LeakyReLU(self.leak)
        ]

        block6 = [
            nn.Conv1d(model_size * 16, model_size * 32, kernel_size, stride, padding, bias=False),
            nn.BatchNorm1d(model_size * 32),
            nn.LeakyReLU(self.leak)
        ]

        all_blocks = block1 + block2 + block3 + block4 + block5 + block6
        self.conv = nn.Sequential(*all_blocks)
        self.linear = nn.Linear(4 * 4 * model_size * 32, 1, bias=False)

        # Initalization with uniform Xavier Glorot technique
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        # Free some memory
        del all_blocks, block1, block2, block3, block4, block5, block6

        # Print summary
        if VERBOSE:
            self.summary()

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