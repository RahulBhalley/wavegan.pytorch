# -*- coding: utf-8 -*-

from __future__ import print_function

__author__ = "Rahul Bhalley"

# Some configuration for networks
SOUND_DIM = 128*128 # [128*128, 256*256]
Z_DIM = 128
BATCH_SIZE = 64
NORM = 'l1' # [l1, l2]
TRAIN = False
RESIDUAL = False

# Some other configurations
DATASET = 'sc09-25-up-pp-xav-uni' # [sc09, birds-songs]
if RESIDUAL:
    DATASET = f'res-{DATASET}'
N_CHANNELS = 1
BEGIN_ITER = 0
TOTAL_ITERS = 100000
ITERS_PER_LOG = 1
VERBOSE = False

print('------------------')
print('| Configurations |')
print('------------------')
print('')
print(f'SOUND_DIM:       {SOUND_DIM}')
print(f'Z_DIM:           {Z_DIM}')
print(f'BATCH_SIZE:      {Z_DIM}')
print(f'NORM:            {Z_DIM}')
print(f'TRAIN:           {TRAIN}')
print(f'DATASET:         {DATASET}')
print(f'N_CHANNELS:      {N_CHANNELS}')
print(f'BEGIN_ITER:      {BEGIN_ITER}')
print(f'TOTAL_ITERS:     {TOTAL_ITERS}')
print(f'ITERS_PER_LOG:   {ITERS_PER_LOG}')
print(f'VERBOSE:         {VERBOSE}')
print('')