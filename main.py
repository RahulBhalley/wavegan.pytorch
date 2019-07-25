# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

__author__ = "Rahul Bhalley"

import torch
import torch.nn as nn
import torch.optim as optim

from data_load import *

#from config import *

if SOUND_DIM == 128 * 128:
    from wavegan_1_024 import Generator, Critic
elif SOUND_DIM == 256 * 256:
    from wavegan_4_096 import Generator, Critic

import os

if TRAIN:
    # Make experiments reproducible
    _ = torch.manual_seed(12345)

####################
# Make directories #
# - Samples        #
# - Checkpoints    #
####################

if not os.path.exists(NORM):
    os.mkdir(NORM)
# Directory for samples
if not os.path.exists(os.path.join(NORM, 'samples', str(SOUND_DIM))):
    os.mkdir(os.path.join(NORM, 'samples', str(SOUND_DIM)))
if not os.path.exists(os.path.join(NORM, 'samples', str(SOUND_DIM), DATASET)):
    os.mkdir(os.path.join(NORM, 'samples', str(SOUND_DIM), DATASET))
# Directory for checkpoints
if not os.path.exists(os.path.join(NORM, 'checkpoints', str(SOUND_DIM))):
    os.mkdir(os.path.join(NORM, 'checkpoints', str(SOUND_DIM)))
if not os.path.exists(os.path.join(NORM, 'checkpoints', str(SOUND_DIM), DATASET)):
    os.mkdir(os.path.join(NORM, 'checkpoints', str(SOUND_DIM), DATASET))
    '''if RESIDUAL and not os.path.exists(os.path.join(NORM, 'checkpoints', str(SOUND_DIM), 'res-' + DATASET)):
        os.mkdir(os.path.exists(os.path.join(NORM, 'checkpoints', str(SOUND_DIM), 'res-' + DATASET)))
'''
####################
# Load the dataset #
####################

def load_data():

    audio_dir = DATASET
    audio_paths = get_all_audio_filepaths(audio_dir)
    train_data, valid_data, test_data, train_size = split_data(audio_path_list=audio_paths, valid_ratio=0.1, test_ratio=0.1, batch_size=BATCH_SIZE)
    TOTAL_TRAIN_SAMPLES = train_size
    BATCH_NUM = TOTAL_TRAIN_SAMPLES // BATCH_SIZE

    # Iterators
    train_iter = iter(train_data)
    valid_iter = iter(valid_data)
    test_iter = iter(test_data)

    return train_iter, valid_iter, test_iter

if TRAIN:
    train_iter, valid_iter, test_iter = load_data()


################################################
# Define device, neural nets, optimizers, etc. #
################################################

# Automatic GPU/CPU device placement
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create models
g_model = Generator().to(device)
c_model = Critic().to(device)

# Optimizers
g_optim = optim.Adam(g_model.parameters(), 2e-4, betas=(0.5, 0.999))
d_optim = optim.Adam(c_model.parameters(), 2e-4, betas=(0.5, 0.999))

############
# Training #
############

def train():
    print('Begin training!')
    # Try loading the latest existing checkpoints based on `BEGIN_ITER`
    try:
        # Checkpoint dirs
        g_model_dir = os.path.join(NORM, 'checkpoints', DATASET, 'g_model_' + str(BEGIN_ITER) + '.pth')
        c_model_dir = os.path.join(NORM, 'checkpoints', DATASET, 'c_model_' + str(BEGIN_ITER) + '.pth')
        # Load checkpoints
        g_model.load_state_dict(torch.load(g_model_dir, map_location='cpu'))
        c_model.load_state_dict(torch.load(c_model_dir, map_location='cpu'))
        print('Loaded the latest checkpoints from {}th iteration.')
        print('NOTE: Set the `BEGIN_ITER` in accordance to saved checkpoints.')
        # Free some memory
        del g_model_dir, c_model_dir
    except:
        print(f"Resume: Couldn't load the checkpoints from {BEGIN_ITER}th iteration.")

    # Just to see the learning progress
    fixed_z = torch.randn(BATCH_SIZE, Z_DIM).to(device)

    for i in range(BEGIN_ITER, TOTAL_ITERS + 1):

        #################
        # Train c_model #
        #################

        # Tune gradient computations
        for param in g_model.parameters():
            param.requires_grad_(False)
        for param in c_model.parameters():
            param.requires_grad_(True)

        for j in range(2):
            print(f"Training c_model: {j}")
            z_sample = torch.randn(BATCH_SIZE, Z_DIM).to(device) # Sample prior from Gaussian distribution
            #x_sample = safe_sampling()
            x_sample = numpy_to_var(train_iter.next()['X'], device)

            with torch.no_grad():
                x_fake = g_model(z_sample)
            x_real_score = c_model(x_sample)
            x_fake_score = c_model(x_fake)

            # Zerofy the gradients
            d_optim.zero_grad()

            print(f'x_sample: {x_sample.size()}')
            print(f'x_fake: {x_fake.size()}')

            # Compute loss
            d_loss = x_real_score - x_fake_score
            if NORM == 'l1':
                d_norm = 10 * (x_sample - x_fake).abs().mean()
            elif NORM == 'l2':
                d_norm = 10 * ((x_sample - x_fake)**2).mean().sqrt()
            d_loss = - d_loss + 0.5 * d_loss**2 / d_norm
            d_loss = d_loss.mean()

            # Compute gradients
            d_loss.backward()

            # Update the network(s)
            d_optim.step()

        #################
        # Train g_model #
        #################

        # Tune gradient computations
        for param in g_model.parameters():
            param.requires_grad_(True)
        for param in c_model.parameters():
            param.requires_grad_(False)
        
        for j in range(1):
            print(f"Training g_model: {j}")
            z_sample = torch.randn(BATCH_SIZE, Z_DIM).to(device) # Sample prior from Gaussian distribution
            #x_sample = safe_sampling()
            x_sample = numpy_to_var(train_iter.next()['X'], device)

            x_fake = g_model(z_sample)
            x_real_score = c_model(x_sample)
            x_fake_score = c_model(x_fake)

            # Zerofy the gradients
            g_optim.zero_grad()

            # Compute loss
            g_loss = x_real_score - x_fake_score
            g_loss = g_loss.mean()

            # Compute gradients
            g_loss.backward()
            
            # Update the network(s)
            g_optim.step()

        ##################
        # Log statistics #
        ##################

        if i % ITERS_PER_LOG == 0:
            # Print statistics
            print(f'iter: {i}, d_loss: {d_loss}, g_loss: {g_loss}')
            # Save image grids of fake and real images
            with torch.no_grad():
                samples = g_model(fixed_z)
            samples = samples.cpu()
            samples = samples.data.numpy()
            samples_dir = os.path.join(NORM, 'samples', str(SOUND_DIM), DATASET)
            save_samples(samples, i, samples_dir)
            print('saved samples')
            # Checkpoint directories
            g_model_dir = os.path.join(NORM, 'checkpoints', str(SOUND_DIM), DATASET, f'g_model_{i}.pth')
            c_model_dir = os.path.join(NORM, 'checkpoints', str(SOUND_DIM), DATASET, f'c_model_{i}.pth')
            # Save all the checkpoints
            torch.save(g_model.state_dict(), g_model_dir)
            torch.save(c_model.state_dict(), c_model_dir)
            print('saved checkpoints')

def infer(epoch, n=10):
    try:
        g_model_dir = os.path.join(NORM, 'checkpoints', str(SOUND_DIM), DATASET, 'g_model_' + str(epoch) + '.pth')
        g_model.load_state_dict(torch.load(g_model_dir, map_location='cpu'))
    except:
        print("Couldn't load the checkpoint of `g_model`.")

    g_model.eval()
    for i in range(n):
        with torch.no_grad():
            z_sample = torch.randn(1, Z_DIM).to(device) # Sample prior from Gaussian distribution
            samples = g_model(z_sample)
        samples = samples.cpu()
        samples = samples.data.numpy()
        samples_dir = os.path.join(NORM, 'samples', str(SOUND_DIM), DATASET)
        save_samples(samples, i, samples_dir)
        print(f'saved samples: {samples_dir}')


#################
# Interpolation #
#################

def interpolate(epoch, mode='lerp', n_latents=1):
    try:
        g_model_dir = os.path.join(NORM, 'checkpoints', str(SOUND_DIM), DATASET, 'g_model_' + str(epoch) + '.pth')
        g_model.load_state_dict(torch.load(g_model_dir, map_location='cpu'))
    except:
        print("Couldn't load the checkpoint of `g_model`.")

    if mode == 'lerp':
        z_start = torch.randn(1, Z_DIM).to(device)
        for a in range(n_latents):
            z_end = torch.randn(1, Z_DIM).to(device)
            z_saver = z_end
            
            # Perform interpolation
            b = 0
            g_model.eval()
            for i in torch.arange(0., 1.0, 0.1):
                with torch.no_grad():
                    z_point = torch.lerp(z_start, z_end, i.item())
                    sample = g_model(z_point)
                sample = sample.cpu()
                sample = sample.data.numpy()
                samples_dir = os.path.join(NORM, 'samples', str(SOUND_DIM), DATASET)
                save_samples(sample, i, samples_dir)
                print(f'Saved sound: {samples_dir}')
                b += 1
            a += 1
            z_start = z_saver

if TRAIN:
    # Train the GAN-QP
    train()
else:
    # Sample from the GAN-QP
    infer(epoch=102000, n=10)
    #interpolate(epoch=102000, mode='lerp', n_latents=5)