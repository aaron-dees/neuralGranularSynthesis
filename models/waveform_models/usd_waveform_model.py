import sys
sys.path.append('../')

from utils.utilities import sample_from_distribution
from scripts.configs.hyper_parameters_waveform import BATCH_SIZE, DEVICE, LATENT_SIZE
from utils.dsp_components import noise_filtering, mod_sigmoid


import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np

# ------------
# Waveform Model Components
# ------------

class stride_conv(nn.Module):
    def __init__(self,kernel_size,in_channels,out_channels,stride):
        super(stride_conv, self).__init__()
        # kernel should be an odd number and stride an even number
        self.conv = nn.Sequential(nn.ReflectionPad1d(kernel_size//2),
                                  nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
                                  nn.BatchNorm1d(out_channels),nn.LeakyReLU(0.2))
    
    def forward(self, x):
        # input and output of shape [bs,in_channels,L] --> [bs,out_channels,L//stride]
        return self.conv(x)


class residual_conv(nn.Module):
    def __init__(self, channels,n_blocks=3):
        super(residual_conv, self).__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.ReflectionPad1d(3**i),
                nn.Conv1d(channels, channels, kernel_size=3, dilation=3**i),
                nn.BatchNorm1d(channels),
                nn.LeakyReLU(0.2),
                nn.Conv1d(channels, channels, kernel_size=1),
                nn.BatchNorm1d(channels))
        for i in range(n_blocks)])
        self.shortcuts = nn.ModuleList([
            nn.Sequential(nn.Conv1d(channels, channels, kernel_size=1),
                      nn.BatchNorm1d(channels))
        for i in range(n_blocks)])
    
    def forward(self, x):
        # input and output of shape [bs,channels,L]
        for block, shortcut in zip(self.blocks, self.shortcuts):
            x = shortcut(x) + block(x)
        return x


class linear_block(nn.Module):
    def __init__(self, in_size,out_size,norm="BN"):
        super(linear_block, self).__init__()
        if norm=="BN":
            self.block = nn.Sequential(nn.Linear(in_size,out_size),nn.BatchNorm1d(out_size),nn.LeakyReLU(0.2))
        if norm=="LN":
            self.block = nn.Sequential(nn.Linear(in_size,out_size),nn.LayerNorm(out_size),nn.LeakyReLU(0.2))
    def forward(self, x):
        return self.block(x)

#############
# Models
#############

class WaveformEncoder(nn.Module):

    def __init__(self,
                    kernel_size=9,
                    channels=128,
                    stride=4,
                    n_convs=3,
                    num_samples=2048,
                    h_dim=512,
                    z_dim=128
                    ):
        super(WaveformEncoder, self).__init__()

        encoder_convs = [nn.Sequential(stride_conv(kernel_size,1,channels,stride),residual_conv(channels,n_blocks=3))]
        encoder_convs += [nn.Sequential(stride_conv(kernel_size,channels,channels,stride),residual_conv(channels,n_blocks=3)) for i in range(1,n_convs)]
        self.encoder_convs = nn.ModuleList(encoder_convs)
         
        self.flatten_size = int(channels*num_samples/(stride**n_convs))
        self.encoder_linears = nn.Sequential(linear_block(self.flatten_size,h_dim),linear_block(h_dim,z_dim))
        self.mu = nn.Linear(z_dim,z_dim)
        self.logvar = nn.Sequential(nn.Linear(z_dim,z_dim),nn.Hardtanh(min_val=-5.0, max_val=5.0)) # clipping to avoid numerical instabilities

    def encode(self, x):

        conv_x = x

        # Convolutional layers
        # x --> h
        for conv in self.encoder_convs:
            conv_x = conv(conv_x)

        # flatten??
        conv_x = conv_x.view(-1,self.flatten_size)

        # Linear layer
        h = self.encoder_linears(conv_x)

        # h --> z
        # h of shape [bs*n_grains,z_dim]
        mu = self.mu(h)
        logvar = self.logvar(h)

        z = sample_from_distribution(mu, logvar)
        return z, mu, logvar

    def forward(self, audio):

        z, mu, logvar = self.encode(audio)

        return z, mu, logvar

class WaveformDecoder(nn.Module):

    def __init__(self,
                    n_linears=3,
                    num_samples = 2048,
                    h_dim=512,
                    z_dim=128
                    ):
        super(WaveformDecoder, self).__init__()

        self.num_samples = num_samples
        self.filter_size = num_samples//2+1

        decoder_linears = [linear_block(z_dim,h_dim)]
        decoder_linears += [linear_block(h_dim,h_dim) for i in range(1,n_linears)]
        decoder_linears += [nn.Linear(h_dim,self.filter_size)]
        self.decoder_linears = nn.Sequential(*decoder_linears)

        self.filter_window = nn.Parameter(torch.fft.fftshift(torch.hann_window(num_samples)),requires_grad=False).to(DEVICE)


    def decode(self, z):

        filter_coeffs = self.decoder_linears(z)

        # What does this do??
        filter_coeffs = mod_sigmoid(filter_coeffs)

        audio = noise_filtering(filter_coeffs, self.filter_window)

        # use below if we were using grains
        # audio = audio.reshape(-1,n_grains,self.hparams.l_grain)
        audio = audio.reshape(-1,1,self.num_samples)

        return audio

    def forward(self, z):

        audio = self.decode(z)

        return audio

class WaveformVAE(nn.Module):

    def __init__(self,
                    kernel_size=9,
                    channels=128,
                    stride=4,
                    n_convs=3,
                    n_linears = 3,
                    num_samples=2048,
                    h_dim=512,
                    z_dim=128
                    ):
        super(WaveformVAE, self).__init__()

        # Encoder and decoder components
        self.Encoder = WaveformEncoder(kernel_size=kernel_size,
                    channels=channels,
                    stride=stride,
                    n_convs=n_convs,
                    num_samples=num_samples,
                    h_dim=h_dim,
                    z_dim=z_dim)
        self.Decoder = WaveformDecoder(n_linears=n_linears,
                    num_samples = num_samples,
                    h_dim=h_dim,
                    z_dim=z_dim)

        # Number of convolutional layers

    def forward(self, x, sampling=True):

        # x ---> z
        z, mu, log_variance = self.Encoder(x);

        # z ---> x_hat
        # Note in paper they also have option passing mu into the decoder and not z
        if sampling:
            x_hat = self.Decoder(z)
        else:
            x_hat = self.Decoder(mu)

        return x_hat, z, mu, log_variance