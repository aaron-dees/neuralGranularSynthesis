import sys
sys.path.append('../')

from utils.utilities import sample_from_distribution, generate_noise_grains
from scripts.configs.hyper_parameters_waveform import BATCH_SIZE, DEVICE, LATENT_SIZE
from utils.dsp_components import noise_filtering, mod_sigmoid


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
    
class MLP(nn.Module):
    """
    MLP (Multi-layer Perception). 

    One layer consists of what as below:
        - 1 Dense Layer
        - 1 Layer Norm
        - 1 ReLU

    constructor arguments :
        n_input : dimension of input
        n_units : dimension of hidden unit
        n_layer : depth of MLP (the number of layers)
        relu : relu (default : nn.ReLU, can be changed to nn.LeakyReLU, nn.PReLU for example.)

    input(x): torch.tensor w/ shape(B, ... , n_input)
    output(x): torch.tensor w/ (B, ..., n_units)
    """

    def __init__(self, n_input, n_units, n_layer, relu=nn.ReLU, inplace=True):
        super(MLP).__init__()
        self.n_layer = n_layer
        self.n_input = n_input
        self.n_units = n_units
        self.inplace = inplace

        self.add_module(
            f"mlp_layer1",
            nn.Sequential(
                nn.Linear(n_input, n_units),
                nn.LayerNorm(normalized_shape=n_units),
                relu(inplace=self.inplace),
            ),
        )

        for i in range(2, n_layer+1):
            self.add_module(
                f"mlp_layer{i}",
                nn.Sequential(
                    nn.Linear(n_units, n_units),
                    nn.LayerNorm(normalized_shape=n_units),
                    relu(inplace=self.inplace),
                ),
            )
        

        mlp_layers = [nn.Sequential(nn.Linear(self.n_input,self.n_units),nn.LayerNorm(self.n_units),relu(inplace=self.inplace))]
        mlp_layers += [nn.Sequential(nn.Linear(self.n_units,self.n_units),nn.LayerNorm(self.n_units),relu(inplace=self.inplace)) for i in range(1, n_layer)]
        self.mlp_layers = nn.Sequential(*mlp_layers)

    def forward(self, x):

        for i in range(1, self.n_layer+1):
            x = self.__getattr__(f"mlp_layer{i}")(x)

        return x


#############
# Models
#############
    
# Waveform encoder is a little similar to wavenet architecture with some changes
class MelSpecEncoder(nn.Module):

    def __init__(self,
                    n_grains,
                    hop_size,
                    n_cc=30,
                    hidden_size=512,
                    bidirectional=False,
                    z_dim = 128,
                    l_grain=2048,
                    n_mels = 128
                    ):
        super(MelSpecEncoder, self).__init__()

        print("Num grains: ", n_grains)
        self.n_grains = n_grains
        self.l_grain = l_grain
        # Set the target length based on hardcodded equation (need to understand why this eq is choosen)
        self.tar_l = int((n_grains+3)/4*l_grain)
        # self.fft_size = (l_grain // 2) + 1
        self.n_mels = n_mels
        self.n_cc = n_cc 
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.z_dim = z_dim
        self.hidden_size = hidden_size
        self.hop_size = hop_size
        self.flatten_size = self.n_mels * self.n_grains

        # Flatten
        self.flatten = nn.Flatten()

        # Dense Layer
        self.mu = nn.Linear(self.flatten_size,z_dim)
        self.logvar = nn.Linear(self.flatten_size,z_dim)
        # self.logvar = nn.Sequential(nn.Linear(z_dim,z_dim),nn.Hardtanh(min_val=-5.0, max_val=5.0)) # clipping to avoid numerical instabilities

    def encode(self, x):

        # Unsqueeze the last layer to allow convolutions, trating spectrogram as greyscale image
        # x = x.unsqueeze(-1)

        #Flatten
        z = self.flatten(x)

        #Dense Layers
        mu = self.mu(z)
        logvar = self.logvar(z)

        z = sample_from_distribution(mu, logvar)
 
        return z, mu, logvar

    def forward(self, audio):

        z, mu, logvar = self.encode(audio)

        return z, mu, logvar

# Waveform decoder consists simply of dense layers.
class MelSpecDecoder(nn.Module):

    def __init__(self,
                    n_grains,
                    hop_size,
                    normalize_ola,
                    # pp_chans,
                    # pp_ker,
                    z_dim,
                    n_mlp_units=512,
                    n_mlp_layers = 3,
                    relu = nn.ReLU,
                    inplace = True,
                    hidden_size = 512,
                    bidirectional = False,
                    n_freq = 513,
                    l_grain = 1024,
                    n_mels = 128,
                    ):
        super(MelSpecDecoder, self).__init__()


        self.n_grains = n_grains
        self.l_grain = l_grain
        self.n_mels = n_mels
        self.tar_l = int((n_grains+3)/4*l_grain)
        self.normalize_ola = normalize_ola
        # self.pp_chans = pp_chans
        self.z_dim = z_dim
        self.n_mlp_units = n_mlp_units
        self.n_mlp_layers = n_mlp_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.n_freq = n_freq
        self.relu = relu
        self.inplace = inplace
        self.hop_size = hop_size

        self.flatten_size = self.n_mels * self.n_grains

        self.dense =  nn.Sequential(nn.Linear(self.z_dim, self.flatten_size), nn.Sigmoid())


    def decode(self, z, n_grains=None, ola_windows=None, ola_folder=None, ola_divisor=None):


        # Step 1
        h = self.dense(z)

        # Rehsape
        h = h.reshape(h.shape[0], self.n_mels, self.n_grains)

        # transform = torchaudio.transforms.GriffinLim(n_fft = 1024, hop_length = self.hop_size, power=2)
        # audio_recon = transform(h)
        
        return h
        # return audio_sum, inv_filter_coeffs.reshape(-1, self.n_grains, inv_filter_coeffs.shape[1]) 

    def forward(self, z, n_grains=None, ola_windows=None, ola_divisor=None):

        audio = self.decode(z, n_grains=n_grains, ola_windows=ola_windows, ola_divisor=ola_divisor)

        return audio
    
class MelSpecVAE(nn.Module):

    def __init__(self,
                    n_grains,
                    hop_size,
                    normalize_ola,
                    n_cc=30,
                    hidden_size=512,
                    bidirectional=False,
                    z_dim = 128,
                    l_grain=1024,                    
                    # pp_chans,
                    # pp_ker,
                    n_mlp_units=512,
                    n_mlp_layers = 3,
                    relu = nn.ReLU,
                    inplace = True,
                    n_freq = 513,
                    n_linears=3,
                    h_dim=512,
                    n_mels = 128
                    ):
        super(MelSpecVAE, self).__init__()

        # Encoder and decoder components
        self.Encoder = MelSpecEncoder(
                        n_grains = n_grains,
                        hop_size = hop_size,
                        n_cc = n_cc,
                        hidden_size = hidden_size,
                        bidirectional = bidirectional,
                        z_dim = z_dim,
                        l_grain = l_grain,
                        n_mels=n_mels,
                    )
        self.Decoder = MelSpecDecoder(
                        n_grains = n_grains,
                        hop_size = hop_size,
                        normalize_ola = normalize_ola,
                        # pp_chans,
                        # pp_ker,
                        z_dim = z_dim,
                        n_mlp_units = n_mlp_units,
                        n_mlp_layers = n_mlp_layers,
                        relu = relu,
                        inplace = inplace,
                        hidden_size = hidden_size,
                        bidirectional = bidirectional,
                        n_freq = n_freq,
                        l_grain = l_grain,
                        n_mels=n_mels
                    )

        # Number of convolutional layers
    def encode(self, x):

         # x ---> z
        # z= self.Encoder(x);
        z, mu, log_variance = self.Encoder(x);
    
        # return {"z":z} 
        return {"z":z,"mu":mu,"logvar":log_variance} 

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
            x_hat= self.Decoder(mu)

        return x_hat, z, mu, log_variance

#########
# Deep Convolutional based Model
#########

# Waveform encoder is a little similar to wavenet architecture with some changes
class MelSpecConvEncoder(nn.Module):

    def __init__(self,
                    n_grains,
                    hop_size,
                    n_cc=30,
                    hidden_size=512,
                    bidirectional=False,
                    z_dim = 128,
                    l_grain=2048,
                    n_mels = 128
                    ):
        super(MelSpecConvEncoder, self).__init__()

        print("Num grains: ", n_grains)
        self.n_grains = n_grains
        self.l_grain = l_grain
        # Set the target length based on hardcodded equation (need to understand why this eq is choosen)
        self.tar_l = int((n_grains+3)/4*l_grain)
        # self.fft_size = (l_grain // 2) + 1
        self.n_mels = n_mels
        self.n_cc = n_cc 
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.z_dim = z_dim
        self.hidden_size = hidden_size
        self.hop_size = hop_size
        self.flatten_size = self.n_mels * self.n_grains

        # Conv Layers
        self.Conv_E_1 = nn.Conv2d(1, 512, stride=2, kernel_size=3, padding=3//2)
        self.Conv_E_2= nn.Conv2d(512, 256, stride=2, kernel_size=3, padding=3//2)
        self.Conv_E_3= nn.Conv2d(256, 128, stride=2, kernel_size=3, padding=3//2)
        self.Conv_E_4= nn.Conv2d(128, 64, stride=2, kernel_size=3, padding=3//2)
        self.Conv_E_5= nn.Conv2d(64, 32, stride=(2,1), kernel_size=3, padding=3//2)
        # Dense layer based on the striding used in conv layers
        self.Dense_E_1 = nn.Linear(32 * int(256/pow(2,5)) * math.ceil(800/pow(2,4)), LATENT_SIZE)

        self.Flat_E_1 = nn.Flatten()

        self.Norm_E_1 = nn.BatchNorm2d(512)
        self.Norm_E_2 = nn.BatchNorm2d(256)
        self.Norm_E_3 = nn.BatchNorm2d(128)
        self.Norm_E_4 = nn.BatchNorm2d(64)
        self.Norm_E_5 = nn.BatchNorm2d(32)


        self.Act_E = nn.ReLU()

    def encode(self, x):

        # Unsqueeze the last layer to allow convolutions, trating spectrogram as greyscale image
        x = x.unsqueeze(1)

        # Conv layer 1
        Conv_E_1 = self.Conv_E_1(x)
        Act_E_1 = self.Act_E(Conv_E_1)
        Norm_E_1 = self.Norm_E_1(Act_E_1)
        # Conv layer 2
        Conv_E_2 = self.Conv_E_2(Norm_E_1)
        Act_E_2 = self.Act_E(Conv_E_2)
        Norm_E_2 = self.Norm_E_2(Act_E_2)
        # Conv layer 3 
        Conv_E_3 = self.Conv_E_3(Norm_E_2)
        Act_E_3 = self.Act_E(Conv_E_3)
        Norm_E_3 = self.Norm_E_3(Act_E_3)
        # Conv layer 4
        Conv_E_4 = self.Conv_E_4(Norm_E_3)
        Act_E_4 = self.Act_E(Conv_E_4)
        Norm_E_4 = self.Norm_E_4(Act_E_4)
        # Conv layer 5
        Conv_E_5 = self.Conv_E_5(Norm_E_4)
        Act_E_5 = self.Act_E(Conv_E_5)
        Norm_E_5 = self.Norm_E_5(Act_E_5)
        # Dense layer for mu and log variance
        Flat_E_1 = self.Flat_E_1(Norm_E_5)
        mu = self.Dense_E_1(Flat_E_1)
        logvar = self.Dense_E_1(Flat_E_1)

        z = sample_from_distribution(mu, logvar)
 
        return z, mu, logvar

    def forward(self, audio):

        z, mu, logvar = self.encode(audio)

        return z, mu, logvar

# Waveform decoder consists simply of dense layers.
class MelSpecConvDecoder(nn.Module):

    def __init__(self,
                    n_grains,
                    hop_size,
                    normalize_ola,
                    # pp_chans,
                    # pp_ker,
                    z_dim,
                    n_mlp_units=512,
                    n_mlp_layers = 3,
                    relu = nn.ReLU,
                    inplace = True,
                    hidden_size = 512,
                    bidirectional = False,
                    n_freq = 513,
                    l_grain = 1024,
                    n_mels = 128,
                    ):
        super(MelSpecConvDecoder, self).__init__()


        self.n_grains = n_grains
        self.l_grain = l_grain
        self.n_mels = n_mels
        self.tar_l = int((n_grains+3)/4*l_grain)
        self.normalize_ola = normalize_ola
        # self.pp_chans = pp_chans
        self.z_dim = z_dim
        self.n_mlp_units = n_mlp_units
        self.n_mlp_layers = n_mlp_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.n_freq = n_freq
        self.relu = relu
        self.inplace = inplace
        self.hop_size = hop_size

        self.Dense_D_1 = nn.Linear(LATENT_SIZE, 32 * int(256/pow(2,5)) * math.ceil(800/pow(2,4)))
        self.ConvT_D_1 = nn.ConvTranspose2d(32, 32, stride=(2,1), kernel_size=3, padding = 3//2, output_padding = (1,0))
        # Note the need to add output padding here for tranposed dimensions to match
        self.ConvT_D_2 = nn.ConvTranspose2d(32, 64, stride=2, kernel_size=3, padding = 3//2, output_padding = 1)
        self.ConvT_D_3 = nn.ConvTranspose2d(64, 128, stride=2, kernel_size=3, padding = 3//2, output_padding = 1)
        self.ConvT_D_4 = nn.ConvTranspose2d(128, 256, stride=2, kernel_size=3, padding = 3//2, output_padding = 1)
        self.ConvT_D_5 = nn.ConvTranspose2d(256, 1, stride=2, kernel_size=3, padding = 3//2, output_padding = 1)
        
        self.Norm_D_1 = nn.BatchNorm2d(32)
        self.Norm_D_2 = nn.BatchNorm2d(64)
        self.Norm_D_3 = nn.BatchNorm2d(128)
        self.Norm_D_4 = nn.BatchNorm2d(256)

        self.Act_D_1 = nn.ReLU()
        self.Act_D_2 = nn.Sigmoid()


    def decode(self, z, n_grains=None, ola_windows=None, ola_folder=None, ola_divisor=None):

        # Dense Layer
        Dense_D_1 = self.Dense_D_1(z)

        # TODO: Find a nicer way of doing this programatically
        # Reshape based on the number striding used in encoder
        Reshape_D_1 = torch.reshape(Dense_D_1, (BATCH_SIZE, 32, int(256/pow(2,5)), math.ceil(800/pow(2,4))))
        # Conv layer 1
        Conv_D_1 = self.ConvT_D_1(Reshape_D_1)
        Act_D_1 = self.Act_D_1(Conv_D_1)
        Norm_D_1 = self.Norm_D_1(Act_D_1)
        # Conv layer 2
        Conv_D_2 = self.ConvT_D_2(Norm_D_1)
        Act_D_2 = self.Act_D_1(Conv_D_2)
        Norm_D_2 = self.Norm_D_2(Act_D_2)
        # Conv layer 3
        Conv_D_3 = self.ConvT_D_3(Norm_D_2)
        Act_D_3 = self.Act_D_1(Conv_D_3)
        Norm_D_3 = self.Norm_D_3(Act_D_3)
        # Conv layer 4
        Conv_D_4 = self.ConvT_D_4(Norm_D_3)
        Act_D_4 = self.Act_D_1(Conv_D_4)
        Norm_D_4 = self.Norm_D_4(Act_D_4)
        # Conv layer 5 (output)
        Conv_D_5= self.ConvT_D_5(Norm_D_4)
        x_hat = self.Act_D_2(Conv_D_5)

        # Remove the additional column
        x_hat = x_hat.reshape(x_hat.shape[0], x_hat.shape[2], x_hat.shape[3])

        return x_hat

    def forward(self, z, n_grains=None, ola_windows=None, ola_divisor=None):

        audio = self.decode(z, n_grains=n_grains, ola_windows=ola_windows, ola_divisor=ola_divisor)

        return audio
    
class MelSpecConvVAE(nn.Module):

    def __init__(self,
                    n_grains,
                    hop_size,
                    normalize_ola,
                    n_cc=30,
                    hidden_size=512,
                    bidirectional=False,
                    z_dim = 128,
                    l_grain=1024,                    
                    # pp_chans,
                    # pp_ker,
                    n_mlp_units=512,
                    n_mlp_layers = 3,
                    relu = nn.ReLU,
                    inplace = True,
                    n_freq = 513,
                    n_linears=3,
                    h_dim=512,
                    n_mels = 128
                    ):
        super(MelSpecConvVAE, self).__init__()

        # Encoder and decoder components
        self.Encoder = MelSpecConvEncoder(
                        n_grains = n_grains,
                        hop_size = hop_size,
                        n_cc = n_cc,
                        hidden_size = hidden_size,
                        bidirectional = bidirectional,
                        z_dim = z_dim,
                        l_grain = l_grain,
                        n_mels=n_mels,
                    )
        self.Decoder = MelSpecConvDecoder(
                        n_grains = n_grains,
                        hop_size = hop_size,
                        normalize_ola = normalize_ola,
                        # pp_chans,
                        # pp_ker,
                        z_dim = z_dim,
                        n_mlp_units = n_mlp_units,
                        n_mlp_layers = n_mlp_layers,
                        relu = relu,
                        inplace = inplace,
                        hidden_size = hidden_size,
                        bidirectional = bidirectional,
                        n_freq = n_freq,
                        l_grain = l_grain,
                        n_mels=n_mels
                    )

        # Number of convolutional layers
    def encode(self, x):

         # x ---> z
        # z= self.Encoder(x);
        z, mu, log_variance = self.Encoder(x);
    
        # return {"z":z} 
        return {"z":z,"mu":mu,"logvar":log_variance} 

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
            x_hat= self.Decoder(mu)

        return x_hat, z, mu, log_variance
