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
import math

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
# Using 3 version based on work in latent timbre synthesis.
#############

#############
# v1
#   - single dense layer
#############

class RISpecEncoder_v1(nn.Module):

    def __init__(self,
                    z_dim = 128,
                    l_grain=2048,
                    h_dim = 512
                    ):
        super(RISpecEncoder_v1, self).__init__()


        self.l_grain = l_grain
        self.h_dim = h_dim
        self.z_dim = z_dim
        kernel_size = 3
        stride = 2
        channels = 128
        n_convs = 3


        # encoder_convs = [nn.Sequential(stride_conv(kernel_size,1,channels,stride),residual_conv(channels,n_blocks=3))]
        # encoder_convs += [nn.Sequential(stride_conv(kernel_size,channels,channels,stride),residual_conv(channels,n_blocks=3)) for i in range(1,n_convs)]
        # self.encoder_convs = nn.ModuleList(encoder_convs)

        self.filter_size = (int((l_grain//2)+1))
        self.encoder_linears = nn.Sequential(linear_block(self.filter_size,h_dim))
        self.mu = nn.Linear(h_dim,z_dim)
        self.logvar = nn.Sequential(nn.Linear(h_dim,z_dim),nn.Hardtanh(min_val=-5.0, max_val=5.0)) # clipping to avoid numerical instabilities
        # self.flatten_size = int(channels*(self.filter_size/(stride**n_convs)+1))
        # self.encoder_linears = nn.Sequential(linear_block(self.flatten_size,h_dim),linear_block(h_dim,z_dim))
        # self.mu = nn.Linear(z_dim,z_dim)
        # self.logvar = nn.Sequential(nn.Linear(z_dim,z_dim),nn.Hardtanh(min_val=-5.0, max_val=5.0)) # clipping to avoid numerical instabilities


    def encode(self, x):

        # The reshape is important for the KL Loss and trating each grains as a batch value,
        # This reshap can be performed here or simply before the KL loss calculation.
        # mb_grains = x.reshape(x.shape[0]*self.n_grains,self.filter_size)

        # # Convolutional layers - feature extraction
        # x = x.unsqueeze(1)
        # # x --> h
        # print(x.shape)
        # for conv in self.encoder_convs:
        #     x = conv(x)
        #     print("output conv size",x.shape)

        # # flatten??
        # print(self.flatten_size)
        # x= x.view(-1,self.flatten_size)

        # Linear layer
        h = self.encoder_linears(x)

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
class RISpecDecoder_v1(nn.Module):

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
                    z_dim,
                    l_grain = 2048,
                    n_linears = 3,
                    h_dim = 512
                    ):
        super(RISpecDecoder_v1, self).__init__()

        self.l_grain = l_grain
        self.filter_size = (l_grain//2+1)
        self.z_dim = z_dim
        self.h_dim = h_dim

        decoder_linears = [linear_block(self.z_dim,self.h_dim)]
        # decoder_linears += [linear_block(h_dim,h_dim) for i in range(1,n_linears)]
        decoder_linears += [nn.Linear(self.h_dim, self.filter_size)]
        self.decoder_linears = nn.Sequential(*decoder_linears)
        self.sig = nn.Sigmoid()

    def decode(self, z, n_grains=None, ola_windows=None, ola_folder=None, ola_divisor=None):


        filter_coeffs = self.decoder_linears(z)

        # What does this do??
        filter_coeffs = mod_sigmoid(filter_coeffs)
        # filter_coeffs = self.sig(filter_coeffs)

        # Reshape back into the batch and grains
        # filter_coeffs = filter_coeffs.reshape(-1, self.n_grains, self.filter_size)

        return filter_coeffs

    def forward(self, z, n_grains=None, ola_windows=None, ola_divisor=None):

        audio = self.decode(z, n_grains=n_grains, ola_windows=ola_windows, ola_divisor=ola_divisor)

        return audio
    
class RISpecVAE_v1(nn.Module):

    def __init__(self,
                    l_grain=2048,                    
                    n_linears=3,
                    z_dim = 128,
                    h_dim=512,
                    ):
        super(RISpecVAE_v1, self).__init__()

        self.z_dim = z_dim
        self.l_grain = l_grain

        # Encoder and decoder components
        self.Encoder = RISpecEncoder_v1(
                        l_grain = l_grain,
                        z_dim = z_dim,
                        h_dim = h_dim,
                    )
        self.Decoder = RISpecDecoder_v1(
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
    
# v2 take prev ri spec grain and the spectral shape
class RISpecEncoder_v2(nn.Module):

    def __init__(self,
                    z_dim = 128,
                    l_grain=2048,
                    h_dim = 512
                    ):
        super(RISpecEncoder_v2, self).__init__()


        self.l_grain = l_grain
        self.h_dim = h_dim
        self.z_dim = z_dim

        self.filter_size = (int((l_grain//2)+1))
        
        self.Linear1 = nn.Linear(self.filter_size, h_dim)
        self.Norm1 = nn.BatchNorm1d(h_dim)
        self.Act1 = nn.LeakyReLU(0.2)

        self.LinearMu = nn.Linear(h_dim,z_dim)
        self.LinearLogvar = nn.Linear(h_dim,z_dim)
        self.ActLogvar = nn.Hardtanh(min_val=-5.0, max_val=5.0)

        # Initialise weights
        torch.nn.init.xavier_uniform_(self.Linear1.weight) 
        torch.nn.init.xavier_uniform_(self.LinearMu.weight) 
        torch.nn.init.xavier_uniform_(self.LinearLogvar.weight) 


    def encode(self, x):

        # The reshape is important for the KL Loss and trating each grains as a batch value,
        # This reshap can be performed here or simply before the KL loss calculation.
        # moving out so n_grains is not required. 
        # mb_grains = x.reshape(x.shape[0]*self.n_grains,self.filter_size)

        # print(" - In: ", x.sum().item())

        # Linear layer x -> h
        h = self.Linear1(x)
        h = self.Norm1(h)
        h = self.Act1(h)


        # h --> z
        # h of shape [bs*n_grains,z_dim]
        mu = self.LinearMu(h)
        logvar = self.LinearLogvar(h)
        logvar = self.ActLogvar(logvar)

        # z of shape [bs*n_grains,z_dim]
        z = sample_from_distribution(mu, logvar)

        return z, mu, logvar

    def forward(self, audio):

        z, mu, logvar = self.encode(audio)
        # z = self.encode(audio)

        return z, mu, logvar
        # return z

# Waveform decoder consists simply of dense layers.
class RISpecDecoder_v2(nn.Module):

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
                    z_dim,
                    l_grain = 2048,
                    n_linears = 3,
                    h_dim = 512
                    ):
        super(RISpecDecoder_v2, self).__init__()

        self.l_grain = l_grain
        self.filter_size = (l_grain//2+1)*2
        self.z_dim = z_dim
        self.h_dim = h_dim

        self.Linear1 = nn.Linear(self.z_dim+self.filter_size, h_dim)
        self.Norm1 = nn.BatchNorm1d(h_dim)
        self.Act1 = nn.LeakyReLU(0.2)

        self.Linear2 = nn.Linear(self.h_dim, self.filter_size)

        self.Act2 = nn.Sigmoid()
        # self.Act2 = nn.Hardtanh(min_val=0.0, max_val=1.0)

        # Initialise weights
        torch.nn.init.xavier_uniform_(self.Linear1.weight) 
        torch.nn.init.xavier_uniform_(self.Linear2.weight) 

    def decode(self, z, prev_ri_spec, ola_windows=None, ola_folder=None, ola_divisor=None):

        # Concatonate the prev ri spec and latent point
        z = torch.cat((z, prev_ri_spec), dim=1)

        #z -> h
        h = self.Linear1(z)
        h = self.Norm1(h)
        h = self.Act1(h)

        h = self.Linear2(h)
        # print("Decoder Linear grad: ", self.Linear2.weight.sum())

        # What does this do??
        # h = mod_sigmoid(h)
        h = 2.0 * self.Act2(h) - 1.0
        # h = (2.0 * self.Act2(h) - 1.0) * (1.0 - 1e-7)

        return h

    def forward(self, z, prev_ri_spec, ola_windows=None, ola_divisor=None):

        audio = self.decode(z, prev_ri_spec, ola_windows=ola_windows, ola_divisor=ola_divisor)

        return audio

    
class RISpecVAE_v2(nn.Module):

    def __init__(self,
                    l_grain=2048,                    
                    n_linears=3,
                    z_dim = 128,
                    h_dim=512,
                    ):
        super(RISpecVAE_v2, self).__init__()

        self.z_dim = z_dim
        self.l_grain = l_grain

        # Encoder and decoder components
        self.Encoder = RISpecEncoder_v2(
                        l_grain = l_grain,
                        z_dim = z_dim,
                        h_dim = h_dim,
                    )
        self.Decoder = RISpecDecoder_v2(
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

    def decode(self, z, prev_ri_spec):

        x_hat = self.Decoder(z, prev_ri_spec)
            
        return {"audio":x_hat}

    def forward(self, x, prev_ri_spec, sampling=True):

        # x ---> z
        # print("Encode")
        
        z, mu, log_variance = self.Encoder(x);
        # print("Decode")
        # z ---> x_hat
        # Note in paper they also have option passing mu into the decoder and not z
        if sampling:
            print("Using Zeros")
            z = torch.randn(z.shape)
            x_hat = self.Decoder(z, prev_ri_spec)
        else:
            x_hat = self.Decoder(mu, prev_ri_spec)
        # print(" - x_hat: ", x_hat.sum().item())


        return x_hat, z, mu, log_variance

# v2 take prev ri spec grain and the spectral shape
# multiple dense layers
class RISpecEncoder_v3(nn.Module):

    def __init__(self,
                    z_dim = 128,
                    l_grain=2048,
                    h_dim = [2048, 1024, 512]
                    ):
        super(RISpecEncoder_v3, self).__init__()


        self.l_grain = l_grain
        self.h_dim = h_dim
        self.z_dim = z_dim

        self.filter_size = (int((l_grain//2)+1))
        
        self.Linear1 = nn.Linear(self.filter_size, h_dim[0])
        self.Norm1 = nn.BatchNorm1d(h_dim[0])
        self.Act1 = nn.LeakyReLU(0.2)

        self.Linear2 = nn.Linear(h_dim[0], h_dim[1])
        self.Norm2 = nn.BatchNorm1d(h_dim[1])
        self.Act2 = nn.LeakyReLU(0.2)

        self.Linear3 = nn.Linear(h_dim[1], h_dim[2])
        self.Norm3 = nn.BatchNorm1d(h_dim[2])
        self.Act3 = nn.LeakyReLU(0.2)

        self.LinearMu = nn.Linear(h_dim[2],z_dim)
        self.LinearLogvar = nn.Linear(h_dim[2],z_dim)
        self.ActLogvar = nn.Hardtanh(min_val=-5.0, max_val=5.0)

        # Initialise weights
        torch.nn.init.xavier_uniform_(self.Linear1.weight) 
        torch.nn.init.xavier_uniform_(self.Linear2.weight) 
        torch.nn.init.xavier_uniform_(self.Linear3.weight) 
        torch.nn.init.xavier_uniform_(self.LinearMu.weight) 
        torch.nn.init.xavier_uniform_(self.LinearLogvar.weight) 


    def encode(self, x):

        # The reshape is important for the KL Loss and trating each grains as a batch value,
        # This reshap can be performed here or simply before the KL loss calculation.
        # moving out so n_grains is not required. 
        # mb_grains = x.reshape(x.shape[0]*self.n_grains,self.filter_size)

        # print(" - In: ", x.sum().item())

        # Linear layer 1 x -> h
        h = self.Linear1(x)
        h = self.Norm1(h)
        h = self.Act1(h)

        # Linear layer 2 x -> h
        h = self.Linear2(h)
        h = self.Norm2(h)
        h = self.Act2(h)

        # Linear layer 3 x -> h
        h = self.Linear3(h)
        h = self.Norm3(h)
        h = self.Act3(h)

        # h --> z
        # h of shape [bs*n_grains,z_dim]
        mu = self.LinearMu(h)
        logvar = self.LinearLogvar(h)
        logvar = self.ActLogvar(logvar)

        # z of shape [bs*n_grains,z_dim]
        z = sample_from_distribution(mu, logvar)

        return z, mu, logvar

    def forward(self, audio):

        z, mu, logvar = self.encode(audio)
        # z = self.encode(audio)

        return z, mu, logvar
        # return z

# Waveform decoder consists simply of dense layers.
class RISpecDecoder_v3(nn.Module):

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
                    z_dim,
                    l_grain = 2048,
                    n_linears = 3,
                    h_dim = [2048, 1024, 512]
                    ):
        super(RISpecDecoder_v3, self).__init__()

        self.l_grain = l_grain
        self.filter_size = (l_grain//2+1)*2
        self.z_dim = z_dim
        self.h_dim = h_dim

        self.Linear1 = nn.Linear(self.z_dim+self.filter_size, h_dim[2])
        self.Norm1 = nn.BatchNorm1d(h_dim[2])
        self.Act1 = nn.LeakyReLU(0.2)

        self.Linear2 = nn.Linear(h_dim[2], h_dim[1])
        self.Norm2 = nn.BatchNorm1d(h_dim[1])
        self.Act2 = nn.LeakyReLU(0.2)

        self.Linear3 = nn.Linear(h_dim[1], h_dim[0])
        self.Norm3 = nn.BatchNorm1d(h_dim[0])
        self.Act3 = nn.LeakyReLU(0.2)

        self.Linear4 = nn.Linear(self.h_dim[0], self.filter_size)

        self.Act4 = nn.Sigmoid()
        # self.Act2 = nn.Hardtanh(min_val=0.0, max_val=1.0)

        # Initialise weights
        torch.nn.init.xavier_uniform_(self.Linear1.weight) 
        torch.nn.init.xavier_uniform_(self.Linear2.weight) 
        torch.nn.init.xavier_uniform_(self.Linear3.weight) 
        torch.nn.init.xavier_uniform_(self.Linear4.weight) 

    def decode(self, z, prev_ri_spec, ola_windows=None, ola_folder=None, ola_divisor=None):

        # Concatonate the prev ri spec and latent point
        z = torch.cat((z, prev_ri_spec), dim=1)

        #z -> h
        h = self.Linear1(z)
        h = self.Norm1(h)
        h = self.Act1(h)

        #h -> h
        h = self.Linear2(h)
        h = self.Norm2(h)
        h = self.Act2(h)

        #h -> h
        h = self.Linear3(h)
        h = self.Norm3(h)
        h = self.Act3(h)

        h = self.Linear4(h)
        # print("Decoder Linear grad: ", self.Linear2.weight.sum())

        # What does this do??
        # h = mod_sigmoid(h)
        h = 2.0 * self.Act4(h) - 1.0
        # h = (2.0 * self.Act2(h) - 1.0) * (1.0 - 1e-7)

        return h

    def forward(self, z, prev_ri_spec, ola_windows=None, ola_divisor=None):

        audio = self.decode(z, prev_ri_spec, ola_windows=ola_windows, ola_divisor=ola_divisor)

        return audio

    
class RISpecVAE_v3(nn.Module):

    def __init__(self,
                    l_grain=2048,                    
                    n_linears=3,
                    z_dim = 128,
                    h_dim=[2048, 1024, 512],
                    ):
        super(RISpecVAE_v3, self).__init__()

        self.z_dim = z_dim
        self.l_grain = l_grain

        # Encoder and decoder components
        self.Encoder = RISpecEncoder_v3(
                        l_grain = l_grain,
                        z_dim = z_dim,
                        h_dim = h_dim,
                    )
        self.Decoder = RISpecDecoder_v3(
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

    def decode(self, z, prev_ri_spec):

        x_hat = self.Decoder(z, prev_ri_spec)
            
        return {"audio":x_hat}

    def forward(self, x, prev_ri_spec, sampling=True):

        # x ---> z
        # print("Encode")
        
        z, mu, log_variance = self.Encoder(x);
        # print("Decode")
        # z ---> x_hat
        # Note in paper they also have option passing mu into the decoder and not z
        if sampling:
            x_hat = self.Decoder(z, prev_ri_spec)
        else:
            x_hat = self.Decoder(mu, prev_ri_spec)
        # print(" - x_hat: ", x_hat.sum().item())


        return x_hat, z, mu, log_variance

# Decoder a seed RI spec (prev timestep) from a point in the latent space.
class SeedGenerator(nn.Module):

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
                    z_dim,
                    l_grain = 2048,
                    n_linears = 3,
                    h_dim = 512
                    ):
        super(SeedGenerator, self).__init__()

        self.l_grain = l_grain
        self.filter_size = (l_grain//2+1)*2
        self.z_dim = z_dim
        self.h_dim = h_dim

        self.Linear1 = nn.Linear(self.z_dim, h_dim)
        self.Norm1 = nn.BatchNorm1d(h_dim)
        self.Act1 = nn.LeakyReLU(0.2)

        self.Linear2 = nn.Linear(self.h_dim, self.filter_size)

        self.Act2 = nn.Sigmoid()
        # self.Act2 = nn.Hardtanh(min_val=0.0, max_val=1.0)

        # Initialise weights
        torch.nn.init.xavier_uniform_(self.Linear1.weight) 
        torch.nn.init.xavier_uniform_(self.Linear2.weight) 

    def generateSeed(self, z, ola_windows=None, ola_folder=None, ola_divisor=None):

        #z -> h
        h = self.Linear1(z)
        h = self.Norm1(h)
        h = self.Act1(h)

        h = self.Linear2(h)
        # print("Decoder Linear grad: ", self.Linear2.weight.sum())

        # What does this do??
        # h = mod_sigmoid(h)
        h = 2.0 * self.Act2(h) - 1.0
        # h = (2.0 * self.Act2(h) - 1.0) * (1.0 - 1e-7)

        return h

    def forward(self, z, ola_windows=None, ola_divisor=None):

        prev_ri_spec = self.generateSeed(z, ola_windows=ola_windows, ola_divisor=ola_divisor)

        return prev_ri_spec
