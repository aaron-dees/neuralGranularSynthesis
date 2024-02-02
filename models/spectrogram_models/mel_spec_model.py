import sys
sys.path.append('../')

from utils.utilities import sample_from_distribution, generate_noise_grains
from scripts.configs.hyper_parameters_waveform import BATCH_SIZE 
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

# Look at implementing something like, conv2d and deconv2d as found in https://github.com/marcoppasini/MelGAN-VC/blob/master/MelGAN_VC.ipynb
# Note the use of 

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
                    conv_filters = (512, 256, 128, 64, 32),
                    conv_kernels = (3, 3, 3, 3, 3),
                    conv_strides = (2, 2, 2, 2, (2,1)),
                    conv_paddings = (3//2, 3//2, 3//2, 3//2, 3//2),
                    l_grain=2048,
                    n_mels = 128,
                    relu = nn.ReLU
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
        self.n_conv_layers = len(conv_filters)
        self.conv_filters = conv_filters
        self.relu = relu

        self.conv_layers = []
        for i in range(self.n_conv_layers):
            if (i == 0):
                self.conv_layers.append(nn.Conv2d(1, conv_filters[i], stride=conv_strides[i], kernel_size=conv_kernels[i], padding=conv_paddings[i]))
                self.conv_layers.append(self.relu())
                self.conv_layers.append(nn.BatchNorm2d(conv_filters[i]))

            else:
                self.conv_layers.append(nn.Conv2d(conv_filters[i-1], conv_filters[i], stride=conv_strides[i], kernel_size=conv_kernels[i], padding=conv_paddings[i]))
                self.conv_layers.append(self.relu())
                self.conv_layers.append(nn.BatchNorm2d(conv_filters[i]))
        
        self.Conv_Layers_E = nn.Sequential(*self.conv_layers)

        # Dense layer based on the striding used in conv layers
        self.Dense_E_1 = nn.Linear(self.conv_filters[-1] * int(self.n_mels/pow(2,self.n_conv_layers)) * math.ceil(self.n_grains/pow(2,self.n_conv_layers-1)), self.z_dim)

        self.Flat_E_1 = nn.Flatten()

    def encode(self, x):

        # Unsqueeze the last layer to allow convolutions, trating spectrogram as greyscale image
        x = x.unsqueeze(1)

        Conv_E_1 = self.Conv_Layers_E(x)

        Flat_E_1 = self.Flat_E_1(Conv_E_1)
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
                    conv_filters = (512, 256, 128, 64, 32),
                    conv_kernels = (3, 3, 3, 3, 3),
                    conv_strides = (2, 2, 2, 2, (2,1)),
                    conv_paddings = (3//2, 3//2, 3//2, 3//2, 3//2),
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
        self.n_conv_layers = len(conv_filters)
        self.conv_filters = conv_filters


        self.conv_layers = []
        for i in reversed(range(self.n_conv_layers)):
            if (i == self.n_conv_layers-1):
                # Note the use of slightly different padding on the first
                self.conv_layers.append(nn.ConvTranspose2d(conv_filters[i], conv_filters[i], stride=conv_strides[i], kernel_size=conv_kernels[i], padding=conv_paddings[i], output_padding = (1,0)))
                self.conv_layers.append(self.relu())
                self.conv_layers.append(nn.BatchNorm2d(conv_filters[i]))

            elif (i == 0):
                self.conv_layers.append(nn.ConvTranspose2d(conv_filters[i+1], 1, stride=conv_strides[i], kernel_size=conv_kernels[i], padding=conv_paddings[i], output_padding = 1))
                self.conv_layers.append(nn.Sigmoid())
                
            else:
                self.conv_layers.append(nn.ConvTranspose2d(conv_filters[i+1], conv_filters[i], stride=conv_strides[i], kernel_size=conv_kernels[i], padding=conv_paddings[i], output_padding=1))
                self.conv_layers.append(self.relu())
                self.conv_layers.append(nn.BatchNorm2d(conv_filters[i]))

        self.ConvT_Layers_D = nn.Sequential(*self.conv_layers)

        self.Dense_D_1 = nn.Linear(self.z_dim, self.conv_filters[-1] * int(self.n_mels/pow(2,self.n_conv_layers)) * math.ceil(self.n_grains/pow(2,self.n_conv_layers-1)))

    def decode(self, z, n_grains=None, ola_windows=None, ola_folder=None, ola_divisor=None):

        # Dense Layer
        Dense_D_1 = self.Dense_D_1(z)

        # Reshape based on the number striding used in encoder
        Reshape_D_1 = torch.reshape(Dense_D_1, (Dense_D_1.shape[0], self.conv_filters[-1], int(self.n_mels/pow(2,self.n_conv_layers)), math.ceil(self.n_grains/pow(2,self.n_conv_layers-1))))
        # Conv layers
        x_hat = self.ConvT_Layers_D(Reshape_D_1)

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
                    conv_filters = (512, 256, 128, 64, 32),
                    conv_kernels = (3, 3, 3, 3, 3),
                    conv_strides = (2, 2, 2, 2, (2,1)),
                    conv_paddings = (3//2, 3//2, 3//2, 3//2, 3//2),
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
                        conv_filters = conv_filters,
                        conv_kernels = conv_kernels,
                        conv_strides = conv_strides,
                        conv_paddings = conv_paddings,
                        n_cc = n_cc,
                        hidden_size = hidden_size,
                        bidirectional = bidirectional,
                        z_dim = z_dim,
                        l_grain = l_grain,
                        n_mels=n_mels,
                        relu=relu
                    )
        self.Decoder = MelSpecConvDecoder(
                        n_grains = n_grains,
                        hop_size = hop_size,
                        normalize_ola = normalize_ola,
                        conv_filters = conv_filters,
                        conv_kernels = conv_kernels,
                        conv_strides = conv_strides,
                        conv_paddings = conv_paddings,
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
