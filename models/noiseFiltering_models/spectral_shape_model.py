import sys
sys.path.append('../')

from utils.utilities import sample_from_distribution, generate_noise_grains
from scripts.configs.hyper_parameters_waveform import BATCH_SIZE, DEVICE, LATENT_SIZE
from utils.dsp_components import noise_filtering, mod_sigmoid, safe_log10


import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.nn import functional as F
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy import fft
import torch_dct as dct
import torchaudio
import librosa

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


# Residual network containing blocks and convolutional skip connections
# Model using residual blocks containing dilated convolutions, 
# note it doesn't look like we're using causal convolutions here.
# It may be interesting to add the storing of skip connections like in wavenet?
class residual_conv(nn.Module):
    def __init__(self, channels,n_blocks=3):
        super(residual_conv, self).__init__()
        self.refpad = nn.ReflectionPad1d(3**1)
        # Block
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
        
        # Shortcut 
        self.shortcuts = nn.ModuleList([
            nn.Sequential(nn.Conv1d(channels, channels, kernel_size=1),
                      nn.BatchNorm1d(channels))
        for i in range(n_blocks)])
    
    def forward(self, x):
        # input and output of shape [bs,channels,L]
        for block, shortcut in zip(self.blocks, self.shortcuts):
            x = shortcut(x) + block(x)
            # x = x + block(x)

            
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
    
#############
# v1
#   - single dense layer
#############

class SpectralEncoder_v1(nn.Module):

    def __init__(self,
                    n_grains,
                    z_dim = 128,
                    l_grain=2048,
                    h_dim = 512
                    ):
        super(SpectralEncoder_v1, self).__init__()


        self.n_grains = n_grains
        self.l_grain = l_grain
        self.h_dim = h_dim
        self.z_dim = z_dim

        self.flatten_size = int((l_grain//2)+1)
        self.encoder_linears = nn.Sequential(linear_block(self.flatten_size,h_dim))
        self.mu = nn.Linear(h_dim,z_dim)
        self.logvar = nn.Sequential(nn.Linear(h_dim,z_dim),nn.Hardtanh(min_val=-5.0, max_val=5.0)) # clipping to avoid numerical instabilities


    def encode(self, x):

        # The reshape is important for the KL Loss and trating each grains as a batch value,
        # This reshap can be performed here or simply before the KL loss calculation.
        mb_grains = x.reshape(x.shape[0]*self.n_grains,(self.l_grain//2)+1)

        # Linear layer
        h = self.encoder_linears(mb_grains)

        # h --> z
        # h of shape [bs*n_grains,z_dim]
        mu = self.mu(h)
        logvar = self.logvar(h)

        # z of shape [bs*n_grains,z_dim]
        z = sample_from_distribution(mu, logvar)

        return z, mu, logvar

    def forward(self, audio):

        z, mu, logvar = self.encode(audio)
        # z = self.encode(audio)

        return z, mu, logvar
        # return z

# Waveform decoder consists simply of dense layers.
class SpectralDecoder_v1(nn.Module):

    """
    Decoder.

    Constructor arguments: 
        use_z : (Bool), if True, Decoder will use z as input.
        mlp_units: 512
        mlp_layers: 3
        z_units: 16
        n_harmonics: 101
        n_freq: 65
        gru_units: 512
        bidirectional: False

    input(dict(f0, z(optional), l)) : a dict object which contains key-values below
        f0 : fundamental frequency for each frame. torch.tensor w/ shape(B, time)
        z : (optional) residual information. torch.tensor w/ shape(B, time, z_units)
        loudness : torch.tensor w/ shape(B, time)

        *note dimension of z is not specified in the paper.

    output : a dict object which contains key-values below
        f0 : same as input
        c : torch.tensor w/ shape(B, time, n_harmonics) which satisfies sum(c) == 1
        a : torch.tensor w/ shape(B, time) which satisfies a > 0
        H : noise filter in frequency domain. torch.tensor w/ shape(B, frame_num, filter_coeff_length)
    """

    def __init__(self,
                    n_grains,
                    z_dim,
                    l_grain = 2048,
                    n_linears = 3,
                    h_dim = 512
                    ):
        super(SpectralDecoder_v1, self).__init__()

        self.n_grains = n_grains
        self.l_grain = l_grain
        self.filter_size = l_grain//2+1
        self.z_dim = z_dim
        self.h_dim = h_dim

        decoder_linears = [linear_block(self.z_dim,self.h_dim)]
        # decoder_linears += [linear_block(h_dim,h_dim) for i in range(1,n_linears)]
        decoder_linears += [nn.Linear(self.h_dim, self.filter_size)]
        self.decoder_linears = nn.Sequential(*decoder_linears)

    def decode(self, z, n_grains=None, ola_windows=None, ola_folder=None, ola_divisor=None):


        filter_coeffs = self.decoder_linears(z)

        # What does this do??
        filter_coeffs = mod_sigmoid(filter_coeffs)

        # Reshape back into the batch and grains
        filter_coeffs = filter_coeffs.reshape(-1, self.n_grains, (self.l_grain//2)+1)

        return filter_coeffs

    def forward(self, z, n_grains=None, ola_windows=None, ola_divisor=None):

        audio = self.decode(z, n_grains=n_grains, ola_windows=ola_windows, ola_divisor=ola_divisor)

        return audio
    
class SpectralVAE_v1(nn.Module):

    def __init__(self,
                    n_grains,
                    l_grain=2048,                    
                    n_linears=3,
                    z_dim = 128,
                    h_dim=512,
                    ):
        super(SpectralVAE_v1, self).__init__()

        self.z_dim = z_dim

        # Encoder and decoder components
        self.Encoder = SpectralEncoder_v1(
                        n_grains = n_grains,
                        l_grain = l_grain,
                        z_dim = z_dim,
                        h_dim = h_dim,
                    )
        self.Decoder = SpectralDecoder_v1(
                        n_grains = n_grains,
                        l_grain = l_grain,
                        z_dim = z_dim,
                        h_dim = h_dim,
                        n_linears = n_linears,

                    )

        # Number of convolutional layers
    def encode(self, x):

        # x ---> z
        z, mu, log_variance = self.Encoder(x);
    
        return {"z":z, "mu":mu, "logvar":log_variance} 

    def decode(self, z):
            
        x_hat = self.Decoder(z)
            
        return {"audio":x_hat}

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


#############
# v2
#   - multiple dense layers
#############

class SpectralEncoder_v2(nn.Module):

    def __init__(self,
                    n_grains,
                    z_dim = 128,
                    l_grain=2048,
                    h_dim = [2048, 1024, 512]
                    ):
        super(SpectralEncoder_v2, self).__init__()


        self.n_grains = n_grains
        self.l_grain = l_grain
        self.h_dim = h_dim
        self.z_dim = z_dim

        self.flatten_size = int((l_grain//2)+1)
        self.encoder_linears = nn.Sequential(linear_block(self.flatten_size, self.h_dim[0]), linear_block(self.h_dim[0], self.h_dim[1]), linear_block(self.h_dim[1], self.h_dim[2]))
        self.mu = nn.Linear(h_dim[2],z_dim)
        self.logvar = nn.Sequential(nn.Linear(h_dim[2], z_dim),nn.Hardtanh(min_val=-5.0, max_val=5.0)) # clipping to avoid numerical instabilities


    def encode(self, x):

        # The reshape is important for the KL Loss and trating each grains as a batch value,
        # This reshap can be performed here or simply before the KL loss calculation.
        mb_grains = x.reshape(x.shape[0]*self.n_grains,(self.l_grain//2)+1)

        # Linear layer
        h = self.encoder_linears(mb_grains)

        # h --> z
        # h of shape [bs*n_grains,z_dim]
        mu = self.mu(h)
        logvar = self.logvar(h)

        # z of shape [bs*n_grains,z_dim]
        z = sample_from_distribution(mu, logvar)

        return z, mu, logvar

    def forward(self, audio):

        z, mu, logvar = self.encode(audio)
        # z = self.encode(audio)

        return z, mu, logvar
        # return z

# Waveform decoder consists simply of dense layers.
class SpectralDecoder_v2(nn.Module):

    """
    Decoder.

    Constructor arguments: 
        use_z : (Bool), if True, Decoder will use z as input.
        mlp_units: 512
        mlp_layers: 3
        z_units: 16
        n_harmonics: 101
        n_freq: 65
        gru_units: 512
        bidirectional: False

    input(dict(f0, z(optional), l)) : a dict object which contains key-values below
        f0 : fundamental frequency for each frame. torch.tensor w/ shape(B, time)
        z : (optional) residual information. torch.tensor w/ shape(B, time, z_units)
        loudness : torch.tensor w/ shape(B, time)

        *note dimension of z is not specified in the paper.

    output : a dict object which contains key-values below
        f0 : same as input
        c : torch.tensor w/ shape(B, time, n_harmonics) which satisfies sum(c) == 1
        a : torch.tensor w/ shape(B, time) which satisfies a > 0
        H : noise filter in frequency domain. torch.tensor w/ shape(B, frame_num, filter_coeff_length)
    """

    def __init__(self,
                    n_grains,
                    z_dim,
                    l_grain = 2048,
                    n_linears = 3,
                    h_dim = [2048, 1024, 512],
                    ):
        super(SpectralDecoder_v2, self).__init__()

        self.n_grains = n_grains
        self.l_grain = l_grain
        self.filter_size = l_grain//2+1
        self.z_dim = z_dim
        self.h_dim = h_dim

        # Do linear layers but in reverse
        decoder_linears = [linear_block(self.z_dim,self.h_dim[2]), linear_block(self.h_dim[2], self.h_dim[1]), linear_block(self.h_dim[1], self.h_dim[0])]
        decoder_linears += [nn.Linear(self.h_dim[0], self.filter_size)]
        self.decoder_linears = nn.Sequential(*decoder_linears)

    def decode(self, z, n_grains=None, ola_windows=None, ola_folder=None, ola_divisor=None):


        filter_coeffs = self.decoder_linears(z)

        # What does this do??
        filter_coeffs = mod_sigmoid(filter_coeffs)

        # Reshape back into the batch and grains
        filter_coeffs = filter_coeffs.reshape(-1, self.n_grains, (self.l_grain//2)+1)

        return filter_coeffs

    def forward(self, z, n_grains=None, ola_windows=None, ola_divisor=None):

        audio = self.decode(z, n_grains=n_grains, ola_windows=ola_windows, ola_divisor=ola_divisor)

        return audio
    
class SpectralVAE_v2(nn.Module):

    def __init__(self,
                    n_grains,
                    l_grain=2048,                    
                    n_linears=3,
                    z_dim = 128,
                    h_dim=[2048, 1024, 512],
                    ):
        super(SpectralVAE_v2, self).__init__()

        self.z_dim = z_dim

        # Encoder and decoder components
        self.Encoder = SpectralEncoder_v2(
                        n_grains = n_grains,
                        l_grain = l_grain,
                        z_dim = z_dim,
                        h_dim = h_dim,
                    )
        self.Decoder = SpectralDecoder_v2(
                        n_grains = n_grains,
                        l_grain = l_grain,
                        z_dim = z_dim,
                        h_dim = h_dim,
                        n_linears = n_linears,

                    )

        # Number of convolutional layers
    def encode(self, x):

        # x ---> z
        z, mu, log_variance = self.Encoder(x);
    
        return {"z":z, "mu":mu, "logvar":log_variance} 

    def decode(self, z):
            
        x_hat = self.Decoder(z)
            
        return {"audio":x_hat}

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
    
#############
# v3
#   - multiple dense layers
#    - CNN layer
#############

class SpectralEncoder_v3(nn.Module):

    def __init__(self,
                    n_grains,
                    z_dim = 128,
                    l_grain=2048,
                    h_dim = [2048, 1024, 512],
                    channels = 32,
                    kernel_size = 3,
                    stride = 2,
                    ):
        super(SpectralEncoder_v3, self).__init__()


        self.n_grains = n_grains
        self.l_grain = l_grain
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.encoder_conv = nn.Sequential(nn.Conv2d(1, self.channels, kernel_size=self.kernel_size, stride=self.stride),
                                  nn.BatchNorm1d(self.channels),nn.LeakyReLU(0.2))

        self.flatten_size = int((l_grain//2)+1)
        self.encoder_linears = nn.Sequential(linear_block(self.flatten_size, self.h_dim[0]), linear_block(self.h_dim[0], self.h_dim[1]), linear_block(self.h_dim[1], self.h_dim[2]))
        self.mu = nn.Linear(h_dim[2],z_dim)
        self.logvar = nn.Sequential(nn.Linear(h_dim[2], z_dim),nn.Hardtanh(min_val=-5.0, max_val=5.0)) # clipping to avoid numerical instabilities


    def encode(self, x):

        # The reshape is important for the KL Loss and trating each grains as a batch value,
        # This reshap can be performed here or simply before the KL loss calculation.
        mb_grains = x.reshape(x.shape[0]*self.n_grains,(self.l_grain//2)+1)

        # reshape
        h = mb_grains.reshape(-1, 19, 27).unsqueeze(1)
        h = mb_grains.reshape(-1, 19, 27)
        print("Reshape shape:", h.shape)

        # conv layers
        h = self.encoder_conv(h)
        print("Conv shape:", h.shape)

        # Linear layer
        h = self.encoder_linears(mb_grains)

        # h --> z
        # h of shape [bs*n_grains,z_dim]
        mu = self.mu(h)
        logvar = self.logvar(h)

        # z of shape [bs*n_grains,z_dim]
        z = sample_from_distribution(mu, logvar)

        return z, mu, logvar

    def forward(self, audio):

        z, mu, logvar = self.encode(audio)
        # z = self.encode(audio)

        return z, mu, logvar
        # return z

# Waveform decoder consists simply of dense layers.
class SpectralDecoder_v3(nn.Module):

    """
    Decoder.

    Constructor arguments: 
        use_z : (Bool), if True, Decoder will use z as input.
        mlp_units: 512
        mlp_layers: 3
        z_units: 16
        n_harmonics: 101
        n_freq: 65
        gru_units: 512
        bidirectional: False

    input(dict(f0, z(optional), l)) : a dict object which contains key-values below
        f0 : fundamental frequency for each frame. torch.tensor w/ shape(B, time)
        z : (optional) residual information. torch.tensor w/ shape(B, time, z_units)
        loudness : torch.tensor w/ shape(B, time)

        *note dimension of z is not specified in the paper.

    output : a dict object which contains key-values below
        f0 : same as input
        c : torch.tensor w/ shape(B, time, n_harmonics) which satisfies sum(c) == 1
        a : torch.tensor w/ shape(B, time) which satisfies a > 0
        H : noise filter in frequency domain. torch.tensor w/ shape(B, frame_num, filter_coeff_length)
    """

    def __init__(self,
                    n_grains,
                    z_dim,
                    l_grain = 2048,
                    n_linears = 3,
                    h_dim = [2048, 1024, 512],
                    ):
        super(SpectralDecoder_v3, self).__init__()

        self.n_grains = n_grains
        self.l_grain = l_grain
        self.filter_size = l_grain//2+1
        self.z_dim = z_dim
        self.h_dim = h_dim

        # Do linear layers but in reverse
        decoder_linears = [linear_block(self.z_dim,self.h_dim[2]), linear_block(self.h_dim[2], self.h_dim[1]), linear_block(self.h_dim[1], self.h_dim[0])]
        decoder_linears += [nn.Linear(self.h_dim[0], self.filter_size)]
        self.decoder_linears = nn.Sequential(*decoder_linears)

    def decode(self, z, n_grains=None, ola_windows=None, ola_folder=None, ola_divisor=None):


        filter_coeffs = self.decoder_linears(z)

        # What does this do??
        filter_coeffs = mod_sigmoid(filter_coeffs)

        # Reshape back into the batch and grains
        filter_coeffs = filter_coeffs.reshape(-1, self.n_grains, (self.l_grain//2)+1)

        return filter_coeffs

    def forward(self, z, n_grains=None, ola_windows=None, ola_divisor=None):

        audio = self.decode(z, n_grains=n_grains, ola_windows=ola_windows, ola_divisor=ola_divisor)

        return audio
    
class SpectralVAE_v3(nn.Module):

    def __init__(self,
                    n_grains,
                    l_grain=2048,                    
                    n_linears=3,
                    z_dim = 128,
                    h_dim=[2048, 1024, 512],
                    channels = 32,
                    kernel_size = 3,
                    stride = 2,
                    ):
        super(SpectralVAE_v3, self).__init__()

        # Encoder and decoder components
        self.Encoder = SpectralEncoder_v3(
                        n_grains = n_grains,
                        l_grain = l_grain,
                        z_dim = z_dim,
                        h_dim = h_dim,
                        channels = channels,
                        kernel_size = kernel_size,
                        stride = stride,
                    )
        self.Decoder = SpectralDecoder_v3(
                        n_grains = n_grains,
                        l_grain = l_grain,
                        z_dim = z_dim,
                        h_dim = h_dim,
                        n_linears = n_linears,
                    )

    def encode(self, x):

        # x ---> z
        z, mu, log_variance = self.Encoder(x);
    
        return {"z":z, "mu":mu, "logvar":log_variance} 

    def decode(self, z):
            
        x_hat = self.Decoder(z)
            
        return {"audio":x_hat}

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
