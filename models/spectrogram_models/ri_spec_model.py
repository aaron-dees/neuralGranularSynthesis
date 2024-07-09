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
                    n_grains,
                    z_dim = 128,
                    l_grain=2048,
                    h_dim = 512
                    ):
        super(RISpecEncoder_v1, self).__init__()


        self.n_grains = n_grains
        self.l_grain = l_grain
        self.h_dim = h_dim
        self.z_dim = z_dim

        self.flatten_size = (int((l_grain//2)+1)) * 2
        self.encoder_linears = nn.Sequential(linear_block(self.flatten_size,h_dim))
        self.mu = nn.Linear(h_dim,z_dim)
        self.logvar = nn.Sequential(nn.Linear(h_dim,z_dim),nn.Hardtanh(min_val=-5.0, max_val=5.0)) # clipping to avoid numerical instabilities


    def encode(self, x):

        # The reshape is important for the KL Loss and trating each grains as a batch value,
        # This reshap can be performed here or simply before the KL loss calculation.
        mb_grains = x.reshape(x.shape[0]*self.n_grains,self.flatten_size)

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
                    n_grains,
                    z_dim,
                    l_grain = 2048,
                    n_linears = 3,
                    h_dim = 512
                    ):
        super(RISpecDecoder_v1, self).__init__()

        self.n_grains = n_grains
        self.l_grain = l_grain
        self.filter_size = (l_grain//2+1)*2
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
        # filter_coeffs = mod_sigmoid(filter_coeffs)
        filter_coeffs = 2.0 * self.sig(filter_coeffs) - 1.0

        # Reshape back into the batch and grains
        filter_coeffs = filter_coeffs.reshape(-1, self.n_grains, self.filter_size)

        return filter_coeffs

    def forward(self, z, n_grains=None, ola_windows=None, ola_divisor=None):

        audio = self.decode(z, n_grains=n_grains, ola_windows=ola_windows, ola_divisor=ola_divisor)

        return audio
    
class RISpecVAE_v1(nn.Module):

    def __init__(self,
                    n_grains,
                    l_grain=2048,                    
                    n_linears=3,
                    z_dim = 128,
                    h_dim=512,
                    ):
        super(RISpecVAE_v1, self).__init__()

        self.z_dim = z_dim
        self.n_grains = n_grains
        self.l_grain = l_grain

        # Encoder and decoder components
        self.Encoder = RISpecEncoder_v1(
                        n_grains = n_grains,
                        l_grain = l_grain,
                        z_dim = z_dim,
                        h_dim = h_dim,
                    )
        self.Decoder = RISpecDecoder_v1(
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
#   - single dense layer
#############

class SpectralEncoder_v2(nn.Module):

    def __init__(self,
                    n_grains,
                    z_dim = 128,
                    l_grain=2048,
                    h_dim = 512
                    ):
        super(SpectralEncoder_v2, self).__init__()


        self.n_grains = n_grains
        self.l_grain = l_grain
        self.h_dim = h_dim
        self.z_dim = z_dim

        self.flatten_size = int((l_grain//2)+1)
        self.encoder_linears = nn.Sequential(linear_block(self.flatten_size,h_dim))
        self.mu = nn.Linear(h_dim,z_dim)
        self.logvar = nn.Sequential(nn.Linear(h_dim,z_dim),nn.Hardtanh(min_val=-5.0, max_val=5.0)) # clipping to avoid numerical instabilities

        # self.Conv_E_1 = nn.Conv2d(1, 128, stride=2, kernel_size=3, padding=3//2)
        # self.Conv_E_2= nn.Conv2d(128, 64, stride=2, kernel_size=3, padding=3//2)
        # self.Conv_E_3= nn.Conv2d(64, 32, stride=2, kernel_size=3, padding=3//2)
        # # self.Conv_E_4= nn.Conv2d(128, 64, stride=2, kernel_size=3, padding=3//2)
        # # self.Conv_E_5= nn.Conv2d(64, 32, stride=(2,1), kernel_size=3, padding=3//2)
        # # Dense layer based on the striding used in conv layers
        # self.Dense_E_1 = nn.Linear(32 * (math.ceil(257/pow(2,3))) * (math.ceil(20/pow(2,3))), LATENT_SIZE)

        # self.Flat_E_1 = nn.Flatten()

        # self.Norm_E_1 = nn.BatchNorm2d(128)
        # self.Norm_E_2 = nn.BatchNorm2d(64)
        # self.Norm_E_3 = nn.BatchNorm2d(32)
        # # self.Norm_E_4 = nn.BatchNorm2d(64)
        # # self.Norm_E_5 = nn.BatchNorm2d(32)

        # self.Act_E_1 = nn.ReLU()

        self.Conv_E_1 = nn.Conv2d(1, 512, stride=2, kernel_size=3, padding=3//2)
        self.Conv_E_2= nn.Conv2d(512, 256, stride=2, kernel_size=3, padding=3//2)
        self.Conv_E_3= nn.Conv2d(256, 128, stride=2, kernel_size=3, padding=3//2)
        self.Conv_E_4= nn.Conv2d(128, 64, stride=2, kernel_size=3, padding=3//2)
        self.Conv_E_5= nn.Conv2d(64, 32, stride=(2,1), kernel_size=3, padding=3//2)
        # Dense layer based on the striding used in conv layers
        self.Dense_E_1 = nn.Linear(32 * (math.ceil(257/pow(2,5))) * (math.ceil(20/pow(2,4))), LATENT_SIZE)

        self.Flat_E_1 = nn.Flatten()

        self.Norm_E_1 = nn.BatchNorm2d(512)
        self.Norm_E_2 = nn.BatchNorm2d(256)
        self.Norm_E_3 = nn.BatchNorm2d(128)
        self.Norm_E_4 = nn.BatchNorm2d(64)
        self.Norm_E_5 = nn.BatchNorm2d(32)

        self.Act_E_1 = nn.ReLU()




    def encode(self, x):

        # Conv layer 1
        # print("Encoder Input shape: ", x.shape)
        Conv_E_1 = self.Conv_E_1(x)
        Act_E_1 = self.Act_E_1(Conv_E_1)
        Norm_E_1 = self.Norm_E_1(Act_E_1)
        # print("Conv Layer 1: ", Norm_E_1.shape)
        # Conv layer 2
        Conv_E_2 = self.Conv_E_2(Norm_E_1)
        Act_E_2 = self.Act_E_1(Conv_E_2)
        Norm_E_2 = self.Norm_E_2(Act_E_2)
        # print("Conv Layer 2: ", Norm_E_2.shape)
        # Conv layer 3 
        Conv_E_3 = self.Conv_E_3(Norm_E_2)
        Act_E_3 = self.Act_E_1(Conv_E_3)
        Norm_E_3 = self.Norm_E_3(Act_E_3)
        # print("Conv Layer 3: ", Norm_E_3.shape)
        # Conv layer 4
        Conv_E_4 = self.Conv_E_4(Norm_E_3)
        Act_E_4 = self.Act_E_1(Conv_E_4)
        Norm_E_4 = self.Norm_E_4(Act_E_4)
        # print("Conv Layer 4: ", Norm_E_4.shape)
        # Conv layer 5
        Conv_E_5 = self.Conv_E_5(Norm_E_4)
        Act_E_5 = self.Act_E_1(Conv_E_5)
        Norm_E_5 = self.Norm_E_5(Act_E_5)
        # print("Conv Layer 5: ", Norm_E_5.shape)
        # Dense layer for mu and log variance
        Flat_E_1 = self.Flat_E_1(Norm_E_5)
        mu = self.Dense_E_1(Flat_E_1)
        logvar = self.Dense_E_1(Flat_E_1)

        # z of shape [bs*n_grains,z_dim]
        z = sample_from_distribution(mu, logvar)
        # print("Encoder Output shape: ", z.shape)

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
                    h_dim = 512
                    ):
        super(SpectralDecoder_v2, self).__init__()

        self.n_grains = n_grains
        self.l_grain = l_grain
        self.filter_size = l_grain//2+1
        self.z_dim = z_dim
        self.h_dim = h_dim

        decoder_linears = [linear_block(self.z_dim,self.h_dim)]
        # decoder_linears += [linear_block(h_dim,h_dim) for i in range(1,n_linears)]
        decoder_linears += [nn.Linear(self.h_dim, self.filter_size)]
        self.decoder_linears = nn.Sequential(*decoder_linears)

        # self.Dense_D_1 = nn.Linear(LATENT_SIZE, 32 * (math.ceil(257/pow(2,3))) * (math.ceil(20/pow(2,3))))

        # # Need to gert a deterministic way of dealing with the below, it depends on wether of not the grain size is odd or even. 
        # # self.ConvT_D_1 = nn.ConvTranspose2d(32, 32, stride=2, kernel_size=3, padding = 3//2, output_padding = (1,0))
        # self.ConvT_D_1 = nn.ConvTranspose2d(32, 32, stride=2, kernel_size=3, padding = 3//2, output_padding = 0)
        # self.ConvT_D_2 = nn.ConvTranspose2d(32, 64, stride=2, kernel_size=3, padding = 3//2, output_padding = (0,1))
        # # self.ConvT_D_2 = nn.ConvTranspose2d(32, 64, stride=2, kernel_size=3, padding = 3//2, output_padding = (0,1))
        # self.ConvT_D_3 = nn.ConvTranspose2d(64, 1, stride=2, kernel_size=3, padding = 3//2, output_padding = (0,1))
        # # self.ConvT_D_3 = nn.ConvTranspose2d(64, 1, stride=2, kernel_size=3, padding = 3//2, output_padding = (0,1))
        
        # self.Norm_D_1 = nn.BatchNorm2d(32)
        # self.Norm_D_2 = nn.BatchNorm2d(64)
        # self.Norm_D_3 = nn.BatchNorm2d(128)
        # # self.Norm_D_4 = nn.BatchNorm2d(256)

        # self.Act_D_1 = nn.ReLU()
        # self.Act_D_2 = nn.Sigmoid()

        # Check is calculating this power and cast kills performance
        # self.Dense_D_1 = nn.Linear(LATENT_SIZE, 32 * int(256/pow(2,5)) * int(64/pow(2,4)))
        self.Dense_D_1 = nn.Linear(LATENT_SIZE, 32 * (math.ceil(257/pow(2,5))) * (math.ceil(20/pow(2,4))))
        self.ConvT_D_1 = nn.ConvTranspose2d(32, 32, stride=(2,1), kernel_size=3, padding = 3//2, output_padding = (0,0))
        # Note the need to add output padding here for tranposed dimensions to match
        self.ConvT_D_2 = nn.ConvTranspose2d(32, 64, stride=2, kernel_size=3, padding = 3//2, output_padding = (0,0))
        self.ConvT_D_3 = nn.ConvTranspose2d(64, 128, stride=2, kernel_size=3, padding = 3//2, output_padding = (0,0))
        self.ConvT_D_4 = nn.ConvTranspose2d(128, 256, stride=2, kernel_size=3, padding = 3//2, output_padding = (0,1))
        self.ConvT_D_5 = nn.ConvTranspose2d(256, 1, stride=2, kernel_size=3, padding = 3//2, output_padding = (0,1))
        
        self.Norm_D_1 = nn.BatchNorm2d(32)
        self.Norm_D_2 = nn.BatchNorm2d(64)
        self.Norm_D_3 = nn.BatchNorm2d(128)
        self.Norm_D_4 = nn.BatchNorm2d(256)

        self.Act_D_1 = nn.ReLU()
        self.Act_D_2 = nn.Sigmoid()


    def decode(self, z, n_grains=None, ola_windows=None, ola_folder=None, ola_divisor=None):

        # z ---> x_hat

        # Dense Layer
        # print("Decoder Input: ", z.shape)
        Dense_D_1 = self.Dense_D_1(z)
        # print("Dense 1: ", Dense_D_1.shape)
        # TODO: Find a nicer way of doing this programatically
        # Reshape based on the number striding used in encoder
        Reshape_D_1 = torch.reshape(Dense_D_1, (10, 32, math.ceil(257/pow(2,5)), math.ceil(20/pow(2,4))))
        # print("Reshape 1: ", Reshape_D_1.shape)
        # Conv layer 1
        Conv_D_1 = self.ConvT_D_1(Reshape_D_1)
        Act_D_1 = self.Act_D_1(Conv_D_1)
        Norm_D_1 = self.Norm_D_1(Act_D_1)
        # print("Conv 1: ", Norm_D_1.shape)
        # Conv layer 2
        Conv_D_2 = self.ConvT_D_2(Norm_D_1)
        Act_D_2 = self.Act_D_1(Conv_D_2)
        Norm_D_2 = self.Norm_D_2(Act_D_2)
        # print("Conv 2: ", Norm_D_2.shape)
        # Conv layer 3
        Conv_D_3 = self.ConvT_D_3(Norm_D_2)
        Act_D_3 = self.Act_D_2(Conv_D_3)
        Norm_D_3 = self.Norm_D_3(Act_D_3)
        # print("Conv 3: ", Norm_D_3.shape)
        # Conv layer 4
        Conv_D_4 = self.ConvT_D_4(Norm_D_3)
        Act_D_4 = self.Act_D_1(Conv_D_4)
        Norm_D_4 = self.Norm_D_4(Act_D_4)
        # print("Conv 4: ", Norm_D_4.shape)
        # Conv layer 5 (output)
        Conv_D_5= self.ConvT_D_5(Norm_D_4)
        x_hat = self.Act_D_2(Conv_D_5)
        # print("Conv 5: ", x_hat.shape)

        return x_hat

    def forward(self, z, n_grains=None, ola_windows=None, ola_divisor=None):

        audio = self.decode(z, n_grains=n_grains, ola_windows=ola_windows, ola_divisor=ola_divisor)

        return audio
    
class SpectralVAE_v2(nn.Module):

    def __init__(self,
                    n_grains,
                    l_grain=2048,                    
                    n_linears=3,
                    z_dim = 128,
                    h_dim=512,
                    ):
        super(SpectralVAE_v2, self).__init__()

        self.z_dim = z_dim
        self.n_grains = n_grains
        self.l_grain = l_grain

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
    

# class SpectralVAE_v2(nn.Module):

#     def __init__(self,
#                     n_grains,
#                     l_grain=2048,                    
#                     n_linears=3,
#                     z_dim = 128,
#                     h_dim=[2048, 1024, 512],
#                     ):
#         super(SpectralVAE_v2, self).__init__()

#         self.z_dim = z_dim

#         # Encoder and decoder components
#         self.Encoder = SpectralEncoder_v2(
#                         n_grains = n_grains,
#                         l_grain = l_grain,
#                         z_dim = z_dim,
#                         h_dim = h_dim,
#                     )
#         self.Decoder = SpectralDecoder_v2(
#                         n_grains = n_grains,
#                         l_grain = l_grain,
#                         z_dim = z_dim,
#                         h_dim = h_dim,
#                         n_linears = n_linears,

#                     )

#         # Number of convolutional layers
#     def encode(self, x):

#         # x ---> z
#         z, mu, log_variance = self.Encoder(x);
    
#         return {"z":z, "mu":mu, "logvar":log_variance} 

#     def decode(self, z):
            
#         x_hat = self.Decoder(z)
            
#         return {"audio":x_hat}

#     def forward(self, x, sampling=True):

#         # x ---> z
        
#         z, mu, log_variance = self.Encoder(x);

#         # z ---> x_hat
#         # Note in paper they also have option passing mu into the decoder and not z
#         if sampling:
#             x_hat = self.Decoder(z)
#         else:
#             x_hat = self.Decoder(mu)

#         return x_hat, z, mu, log_variance
    
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
