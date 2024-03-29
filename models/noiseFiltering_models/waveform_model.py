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
class WaveformEncoder(nn.Module):

    def __init__(self,
                    n_grains,
                    hop_size,
                    kernel_size=9,
                    channels=128,
                    stride=4,
                    n_convs=3,
                    num_samples=2048,
                    l_grain=2048,
                    h_dim=512,
                    z_dim=128
                    ):
        super(WaveformEncoder, self).__init__()


        self.n_grains = n_grains
        self.l_grain = l_grain
        # Set the target length based on hardcodded equation (need to understand why this eq is choosen)
        self.tar_l = int((n_grains+3)/4*l_grain)

        # define the slice_kernel, this is used in convolution to expand out the grains.
        # TODO look into this a little more, what is eye, identity matrix?
        self.slice_kernel = nn.Parameter(torch.eye(l_grain).unsqueeze(1),requires_grad=False)

        self.hop_size = hop_size

        # Overlap and Add windows for each grain, first half of first grain has no window (ie all values = 1) and 
        # last half of the last grain has no window (ie all values = 1), to allow preserverance of attack and decay.
        ola_window = signal.hann(l_grain,sym=False)
        ola_windows = torch.from_numpy(ola_window).unsqueeze(0).repeat(n_grains,1).type(torch.float32)
        ola_windows[0,:l_grain//2] = ola_window[l_grain//2] # start of 1st grain is not windowed for preserving attacks
        ola_windows[-1,l_grain//2:] = ola_window[l_grain//2] # end of last grain is not wondowed to preserving decays
        self.ola_windows = nn.Parameter(ola_windows,requires_grad=False)

        encoder_convs = [nn.Sequential(stride_conv(kernel_size,1,channels,stride),residual_conv(channels,n_blocks=3))]
        encoder_convs += [nn.Sequential(stride_conv(kernel_size,channels,channels,stride),residual_conv(channels,n_blocks=3)) for i in range(1,n_convs)]
        # print(encoder_convs)
        self.encoder_convs = nn.ModuleList(encoder_convs)

        self.flatten_size = int(channels*l_grain/(stride**n_convs))
        self.encoder_linears = nn.Sequential(linear_block(self.flatten_size,h_dim),linear_block(h_dim,z_dim))
        self.mu = nn.Linear(z_dim,z_dim)
        self.logvar = nn.Sequential(nn.Linear(z_dim,z_dim),nn.Hardtanh(min_val=-5.0, max_val=5.0)) # clipping to avoid numerical instabilities

    def encode(self, x):

        # print("Running necoder")

        # This turns our sample into overlapping grains
        mb_grains = F.conv1d(x.unsqueeze(1),self.slice_kernel,stride=self.hop_size,groups=1,bias=None)
        mb_grains = mb_grains.permute(0,2,1)
        bs = mb_grains.shape[0]
        # repeat the overlap add windows acrosss the batch and apply to the grains
        mb_grains = mb_grains*(self.ola_windows.unsqueeze(0).repeat(bs,1,1))
        # merge the grain dimension into batch dimension
        # mb_grains of shape [bs*n_grains,1,l_grain]
        mb_grains = mb_grains.reshape(bs*self.n_grains,self.l_grain).unsqueeze(1)

        # Convolutional layers - feature extraction
        # x --> h
        for conv in self.encoder_convs:
            mb_grains = conv(mb_grains)
            # print("output conv size",mb_grains.shape)

        # flatten??
        mb_grains = mb_grains.view(-1,self.flatten_size)

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

        return z, mu, logvar

# Waveform decoder consists simply of dense layers.
class WaveformDecoder(nn.Module):

    def __init__(self,
                    n_grains,
                    hop_size,
                    normalize_ola,
                    pp_chans,
                    pp_ker,
                    n_linears=3,
                    l_grain = 2048,
                    h_dim=512,
                    z_dim=128,
                    hop_ratio=0.25
                    ):
        super(WaveformDecoder, self).__init__()

        self.n_grains = n_grains
        self.l_grain = l_grain
        self.filter_size = l_grain//2+1
        self.tar_l = int((n_grains+3)/4*l_grain)
        self.normalize_ola = normalize_ola
        self.pp_chans = pp_chans
        self.hop_ratio = hop_ratio

        decoder_linears = [linear_block(z_dim,h_dim)]
        decoder_linears += [linear_block(h_dim,h_dim) for i in range(1,n_linears)]
        decoder_linears += [nn.Linear(h_dim,self.filter_size)]
        self.decoder_linears = nn.Sequential(*decoder_linears)

        self.filter_window = nn.Parameter(torch.fft.fftshift(torch.hann_window(l_grain)),requires_grad=False).to(DEVICE)

        # Overlap and Add windows for each grain, first half of first grain has no window (ie all values = 1) and 
        # last half of the last grain has no window (ie all values = 1), to allow preserverance of attack and decay.
        ola_window = signal.hann(l_grain,sym=False)
        ola_windows = torch.from_numpy(ola_window).unsqueeze(0).repeat(n_grains,1).type(torch.float32)
        ola_windows[0,:l_grain//2] = ola_window[l_grain//2] # start of 1st grain is not windowed for preserving attacks
        ola_windows[-1,l_grain//2:] = ola_window[l_grain//2] # end of last grain is not wondowed to preserving decays
        self.ola_windows = nn.Parameter(ola_windows,requires_grad=False)
        print(self.ola_windows.shape)

        # Folder
        # Folds input tensor into shape [bs, channels, tar_l, 1], using a kernel size of l_grain, and stride of hop_size
        # can see doc here, https://pytorch.org/docs/stable/generated/torch.nn.Fold.html
        self.ola_folder = nn.Fold((self.tar_l,1),(l_grain,1),stride=(hop_size,1))

        # Normalize OLA
        # This attempts to normalize the energy by dividing by the number of 
        # overlapping grains used when folding to get each point in times energy (amplitude).
        if normalize_ola:
            unfolder = nn.Unfold((l_grain,1),stride=(hop_size,1))
            input_ones = torch.ones(1,1,self.tar_l,1)
            ola_divisor = self.ola_folder(unfolder(input_ones)).squeeze()
            self.ola_divisor = nn.Parameter(ola_divisor,requires_grad=False)
        
        # TODO Look at NGS paper, and ref paper as to how and why this works.
        self.post_pro = nn.Sequential(nn.Conv1d(pp_chans, 1, pp_ker, padding=pp_ker//2),nn.Softsign())

    def decode(self, z, n_grains=None, ola_windows=None, ola_folder=None, ola_divisor=None):

        filter_coeffs = self.decoder_linears(z)

        # What does this do??
        filter_coeffs = mod_sigmoid(filter_coeffs)
        orig_filter_coeffs = filter_coeffs
        # filter_coeffs = mod_sigmoid(filter_coeffs - 5)

        # CC part
        # filter_coeffs[:, 128:] = 0.0

        # NOTE Do i need to  do the scaling back from decibels, also note this introduces 
        # NOTE is there a torch implementation of this, bit of a bottleneck if not?
        # NOTE issue with gradient flowwing back
        # NOTE changed this from idct_2d to idct
        # inv_filter_coeffs = (dct.idct(filter_coeffs))
        # inv_filter_coeffs = 10**(dct.idct(filter_coeffs) / 20)

        # inv_filter_coeffs = torch.from_numpy(inv_cepstral_coeff, device=filter_coeffs.device)

        # try predicting only the needed coefficients.
        # cc = torch.nn.functional.pad(filter_coeffs, (0, (2048/2 +1) - filter_coeffs.shape[1]))
        
        # audio = noise_filtering(inv_filter_coeffs, self.filter_window, self.n_grains, self.l_grain)
        audio = noise_filtering(orig_filter_coeffs, self.filter_window, self.n_grains, self.l_grain, self.hop_ratio)


        # Check if number of grains wanted is entered, else use the original
        if n_grains is None:
            audio = audio.reshape(-1,self.n_grains,self.l_grain)
        else:
            audio = audio.reshape(-1,n_grains,self.l_grain)
        bs = audio.shape[0]

        # Check if an overlapp add window has been passed, if not use that used in encoding.
        if ola_windows is None:
            audio = audio*(self.ola_windows.unsqueeze(0).repeat(bs,1,1))
        else:
            audio = audio*(ola_windows.unsqueeze(0).repeat(bs,1,1))


        # Overlap add folder, folds and reshapes audio into target dimensions, so [bs, tar_l]
        # This is essentially folding and adding the overlapping grains back into original sample size, based on the given hop size.
        # Note that shape is changed here, so that input tensor is of chape [bs, channel X (kernel_size), L],
        # since kernel size is l_grain, this is needed in the second dimension.
        if ola_folder is None:
            audio_sum = self.ola_folder(audio.permute(0,2,1)).squeeze()
        else:
            audio_sum = ola_folder(audio.permute(0,2,1)).squeeze()

        # Normalise the energy values across the audio samples
        if self.normalize_ola:
            if ola_divisor is None:
                # Normalises based on number of overlapping grains used in folding per point in time.
                audio_sum = audio_sum/self.ola_divisor.unsqueeze(0).repeat(bs,1)
            else:
                audio_sum = audio_sum/ola_divisor.unsqueeze(0).repeat(bs,1)

        # NOTE Removed the post processing step for now
        # This module applies a multi-channel temporal convolution that
        # learns a parallel set of time-invariant FIR filters and improves
        # the audio quality of the assembled signal.
        audio_sum = self.post_pro(audio_sum.unsqueeze(1).repeat(1,self.pp_chans,1)).squeeze(1)

        return audio_sum 

    def forward(self, z, n_grains=None, ola_windows=None, ola_divisor=None):

        audio = self.decode(z, n_grains=n_grains, ola_windows=ola_windows, ola_divisor=ola_divisor)

        return audio

class WaveformVAE(nn.Module):

    def __init__(self,
                    n_grains,
                    hop_size,
                    normalize_ola,
                    pp_chans,
                    pp_ker,
                    kernel_size=9,
                    channels=128,
                    stride=4,
                    n_convs=3,
                    n_linears = 3,
                    num_samples=2048,
                    l_grain = 2048,
                    h_dim=512,
                    z_dim=128
                    ):
        super(WaveformVAE, self).__init__()

        self.z_dim = z_dim
        self.n_grains = n_grains

        # Encoder and decoder components
        self.Encoder = WaveformEncoder(
                    hop_size=hop_size,
                    n_grains=n_grains,
                    kernel_size=kernel_size,
                    channels=channels,
                    stride=stride,
                    n_convs=n_convs,
                    num_samples=num_samples,
                    l_grain=l_grain,
                    h_dim=h_dim,
                    z_dim=z_dim)
        self.Decoder = WaveformDecoder(
                    n_grains=n_grains,
                    hop_size=hop_size,
                    normalize_ola=normalize_ola,
                    pp_chans=pp_chans,
                    pp_ker=pp_ker,
                    n_linears=n_linears,
                    l_grain=l_grain,
                    h_dim=h_dim,
                    z_dim=z_dim)

        # Number of convolutional layers
    def encode(self, x):

         # x ---> z
        z, mu, log_variance = self.Encoder(x);
    
        return {"z":z,"mu":mu,"logvar":log_variance} 

    def decode(self, z):
            
        x_hat = self.Decoder(z)
            
        return x_hat

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

class WaveformEncoderTest(nn.Module):

    def __init__(self,
                    n_grains,
                    hop_size,
                    kernel_size=9,
                    channels=128,
                    stride=4,
                    n_convs=3,
                    num_samples=2048,
                    l_grain=2048,
                    h_dim=512,
                    z_dim=128
                    ):
        super(WaveformEncoderTest, self).__init__()


        self.n_grains = n_grains
        self.l_grain = l_grain
        # Set the target length based on hardcodded equation (need to understand why this eq is choosen)
        self.tar_l = int((n_grains+3)/4*l_grain)

        # define the slice_kernel, this is used in convolution to expand out the grains.
        # TODO look into this a little more, what is eye, identity matrix?
        self.slice_kernel = nn.Parameter(torch.eye(l_grain).unsqueeze(1),requires_grad=False)

        self.hop_size = hop_size

        # Overlap and Add windows for each grain, first half of first grain has no window (ie all values = 1) and 
        # last half of the last grain has no window (ie all values = 1), to allow preserverance of attack and decay.
        ola_window = signal.hann(l_grain,sym=False)
        ola_windows = torch.from_numpy(ola_window).unsqueeze(0).repeat(n_grains,1).type(torch.float32)
        ola_windows[0,:l_grain//2] = ola_window[l_grain//2] # start of 1st grain is not windowed for preserving attacks
        ola_windows[-1,l_grain//2:] = ola_window[l_grain//2] # end of last grain is not wondowed to preserving decays
        self.ola_windows = nn.Parameter(ola_windows,requires_grad=False)

        encoder_convs = [nn.Sequential(stride_conv(kernel_size,1,channels,stride),residual_conv(channels,n_blocks=3))]
        encoder_convs += [nn.Sequential(stride_conv(kernel_size,channels,channels,stride),residual_conv(channels,n_blocks=3)) for i in range(1,n_convs)]
        # print(encoder_convs)
        self.encoder_convs = nn.ModuleList(encoder_convs)

        # self.flatten_size = int(channels*l_grain/(stride**n_convs))
        # self.flatten_size = int(l_grain)
        self.flatten_size = int((l_grain//2)+1)
        self.encoder_linears = nn.Sequential(linear_block(self.flatten_size,h_dim),linear_block(h_dim,z_dim))
        self.mu = nn.Linear(z_dim,z_dim)
        self.logvar = nn.Sequential(nn.Linear(z_dim,z_dim),nn.Hardtanh(min_val=-5.0, max_val=5.0)) # clipping to avoid numerical instabilities

    def encode(self, x):

        # print("Running necoder")

        # This turns our sample into overlapping grains
        # mb_grains = F.conv1d(x.unsqueeze(1),self.slice_kernel,stride=self.hop_size,groups=1,bias=None)
        # mb_grains = mb_grains.permute(0,2,1)
        # bs = mb_grains.shape[0]
        # # repeat the overlap add windows acrosss the batch and apply to the grains
        # mb_grains = mb_grains*(self.ola_windows.unsqueeze(0).repeat(bs,1,1))
        # merge the grain dimension into batch dimension
        # mb_grains of shape [bs*n_grains,1,l_grain]

        # mb_grains = x.reshape(x.shape[0]*self.n_grains,self.l_grain)
        mb_grains = x.reshape(x.shape[0]*self.n_grains,(self.l_grain//2)+1)

        # mb_grains = x.reshape(x.shape[0]*self.n_grains,self.l_grain).unsqueeze(1)

        # print("Input: ", x.shape)

        # Convolutional layers - feature extraction
        # x --> h
        # for conv in self.encoder_convs:
        #     mb_grains = conv(mb_grains)
        #     # print("output conv size",mb_grains.shape)

        # flatten??
        # mb_grains = mb_grains.view(-1,self.flatten_size)
        # mb_grains = mb_grains.reshape(mb_grains.shape[0], self.flatten_size)
        # print("After flatten: ", mb_grains.shape)

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

        return z, mu, logvar

# Waveform decoder consists simply of dense layers.
class WaveformDecoderTest(nn.Module):

    def __init__(self,
                    n_grains,
                    hop_size,
                    normalize_ola,
                    pp_chans,
                    pp_ker,
                    n_linears=3,
                    l_grain = 2048,
                    h_dim=512,
                    z_dim=128,
                    hop_ratio=0.25
                    ):
        super(WaveformDecoderTest, self).__init__()

        self.n_grains = n_grains
        self.l_grain = l_grain
        self.filter_size = l_grain//2+1
        self.tar_l = int((n_grains+3)/4*l_grain)
        self.normalize_ola = normalize_ola
        self.pp_chans = pp_chans
        self.hop_ratio = hop_ratio

        decoder_linears = [linear_block(z_dim,h_dim)]
        decoder_linears += [linear_block(h_dim,h_dim) for i in range(1,n_linears)]
        decoder_linears += [nn.Linear(h_dim,self.filter_size)]
        self.decoder_linears = nn.Sequential(*decoder_linears)

        self.filter_window = nn.Parameter(torch.fft.fftshift(torch.hann_window(l_grain)),requires_grad=False).to(DEVICE)

        # Overlap and Add windows for each grain, first half of first grain has no window (ie all values = 1) and 
        # last half of the last grain has no window (ie all values = 1), to allow preserverance of attack and decay.
        ola_window = signal.hann(l_grain,sym=False)
        ola_windows = torch.from_numpy(ola_window).unsqueeze(0).repeat(n_grains,1).type(torch.float32)
        ola_windows[0,:l_grain//2] = ola_window[l_grain//2] # start of 1st grain is not windowed for preserving attacks
        ola_windows[-1,l_grain//2:] = ola_window[l_grain//2] # end of last grain is not wondowed to preserving decays
        self.ola_windows = nn.Parameter(ola_windows,requires_grad=False)
        print(self.ola_windows.shape)

        # Folder
        # Folds input tensor into shape [bs, channels, tar_l, 1], using a kernel size of l_grain, and stride of hop_size
        # can see doc here, https://pytorch.org/docs/stable/generated/torch.nn.Fold.html
        self.ola_folder = nn.Fold((self.tar_l,1),(l_grain,1),stride=(hop_size,1))

        # Normalize OLA
        # This attempts to normalize the energy by dividing by the number of 
        # overlapping grains used when folding to get each point in times energy (amplitude).
        if normalize_ola:
            unfolder = nn.Unfold((l_grain,1),stride=(hop_size,1))
            input_ones = torch.ones(1,1,self.tar_l,1)
            ola_divisor = self.ola_folder(unfolder(input_ones)).squeeze()
            self.ola_divisor = nn.Parameter(ola_divisor,requires_grad=False)
        
        # TODO Look at NGS paper, and ref paper as to how and why this works.
        self.post_pro = nn.Sequential(nn.Conv1d(pp_chans, 1, pp_ker, padding=pp_ker//2),nn.Softsign())

    def decode(self, z, n_grains=None, ola_windows=None, ola_folder=None, ola_divisor=None):

        filter_coeffs = self.decoder_linears(z)

        # What does this do??
        filter_coeffs = mod_sigmoid(filter_coeffs)
        orig_filter_coeffs = filter_coeffs
        # filter_coeffs = mod_sigmoid(filter_coeffs - 5)

        # CC part
        # filter_coeffs[:, 128:] = 0.0

        # NOTE Do i need to  do the scaling back from decibels, also note this introduces 
        # NOTE is there a torch implementation of this, bit of a bottleneck if not?
        # NOTE issue with gradient flowwing back
        # NOTE changed this from idct_2d to idct
        # inv_filter_coeffs = (dct.idct(filter_coeffs))
        # inv_filter_coeffs = 10**(dct.idct(filter_coeffs) / 20)

        # inv_filter_coeffs = torch.from_numpy(inv_cepstral_coeff, device=filter_coeffs.device)

        # try predicting only the needed coefficients.
        # cc = torch.nn.functional.pad(filter_coeffs, (0, (2048/2 +1) - filter_coeffs.shape[1]))
        
        # audio = noise_filtering(inv_filter_coeffs, self.filter_window, self.n_grains, self.l_grain)
        audio = noise_filtering(orig_filter_coeffs, self.filter_window, self.n_grains, self.l_grain, self.hop_ratio)


        # Check if number of grains wanted is entered, else use the original
        if n_grains is None:
            audio = audio.reshape(-1,self.n_grains,self.l_grain)
        else:
            audio = audio.reshape(-1,n_grains,self.l_grain)
        bs = audio.shape[0]

        # Check if an overlapp add window has been passed, if not use that used in encoding.
        if ola_windows is None:
            audio = audio*(self.ola_windows.unsqueeze(0).repeat(bs,1,1))
        else:
            audio = audio*(ola_windows.unsqueeze(0).repeat(bs,1,1))


        # Overlap add folder, folds and reshapes audio into target dimensions, so [bs, tar_l]
        # This is essentially folding and adding the overlapping grains back into original sample size, based on the given hop size.
        # Note that shape is changed here, so that input tensor is of chape [bs, channel X (kernel_size), L],
        # since kernel size is l_grain, this is needed in the second dimension.
        if ola_folder is None:
            audio_sum = self.ola_folder(audio.permute(0,2,1)).squeeze()
        else:
            audio_sum = ola_folder(audio.permute(0,2,1)).squeeze()

        # Normalise the energy values across the audio samples
        if self.normalize_ola:
            if ola_divisor is None:
                # Normalises based on number of overlapping grains used in folding per point in time.
                audio_sum = audio_sum/self.ola_divisor.unsqueeze(0).repeat(bs,1)
            else:
                audio_sum = audio_sum/ola_divisor.unsqueeze(0).repeat(bs,1)

        # NOTE Removed the post processing step for now
        # This module applies a multi-channel temporal convolution that
        # learns a parallel set of time-invariant FIR filters and improves
        # the audio quality of the assembled signal.
        # audio_sum = self.post_pro(audio_sum.unsqueeze(1).repeat(1,self.pp_chans,1)).squeeze(1)

        return audio_sum 

    def forward(self, z, n_grains=None, ola_windows=None, ola_divisor=None):

        audio = self.decode(z, n_grains=n_grains, ola_windows=ola_windows, ola_divisor=ola_divisor)

        return audio

class WaveformVAETest(nn.Module):

    def __init__(self,
                    n_grains,
                    hop_size,
                    normalize_ola,
                    pp_chans,
                    pp_ker,
                    kernel_size=9,
                    channels=128,
                    stride=4,
                    n_convs=3,
                    n_linears = 3,
                    num_samples=2048,
                    l_grain = 2048,
                    h_dim=512,
                    z_dim=128
                    ):
        super(WaveformVAETest, self).__init__()

        self.z_dim = z_dim
        self.n_grains = n_grains

        # Encoder and decoder components
        self.Encoder = WaveformEncoderTest(
                    hop_size=hop_size,
                    n_grains=n_grains,
                    kernel_size=kernel_size,
                    channels=channels,
                    stride=stride,
                    n_convs=n_convs,
                    num_samples=num_samples,
                    l_grain=l_grain,
                    h_dim=h_dim,
                    z_dim=z_dim)
        self.Decoder = WaveformDecoderTest(
                    n_grains=n_grains,
                    hop_size=hop_size,
                    normalize_ola=normalize_ola,
                    pp_chans=pp_chans,
                    pp_ker=pp_ker,
                    n_linears=n_linears,
                    l_grain=l_grain,
                    h_dim=h_dim,
                    z_dim=z_dim)

        # Number of convolutional layers
    def encode(self, x):

         # x ---> z
        z, mu, log_variance = self.Encoder(x);
    
        return {"z":z,"mu":mu,"logvar":log_variance} 

    def decode(self, z):
            
        x_hat = self.Decoder(z)
            
        return x_hat

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
    
# Waveform encoder is a little similar to wavenet architecture with some changes
class CepstralCoeffsEncoder(nn.Module):

    def __init__(self,
                    n_grains,
                    hop_size,
                    n_cc=30,
                    hidden_size=512,
                    bidirectional=False,
                    z_dim = 128,
                    l_grain=2048,
                    ):
        super(CepstralCoeffsEncoder, self).__init__()


        self.n_grains = n_grains
        self.l_grain = l_grain
        # Set the target length based on hardcodded equation (need to understand why this eq is choosen)
        self.tar_l = int((n_grains+3)/4*l_grain)

        self.n_cc = n_cc
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.z_dim = z_dim
        self.hidden_size = hidden_size

        # define the slice_kernel, this is used in convolution to expand out the grains.
        # TODO look into this a little more, what is eye, identity matrix?
        self.slice_kernel = nn.Parameter(torch.eye(l_grain).unsqueeze(1),requires_grad=False)

        self.hop_size = hop_size

        # Overlap and Add windows for each grain, first half of first grain has no window (ie all values = 1) and 
        # last half of the last grain has no window (ie all values = 1), to allow preserverance of attack and decay.
        ola_window = signal.hann(l_grain,sym=False)
        ola_windows = torch.from_numpy(ola_window).unsqueeze(0).repeat(n_grains,1).type(torch.float32)
        ola_windows[0,:l_grain//2] = ola_window[l_grain//2] # start of 1st grain is not windowed for preserving attacks
        ola_windows[-1,l_grain//2:] = ola_window[l_grain//2] # end of last grain is not wondowed to preserving decays
        self.ola_windows = nn.Parameter(ola_windows,requires_grad=False)


        # self.flatten_size = int(channels*l_grain/(stride**n_convs))
        # self.encoder_linears = nn.Sequential(linear_block(self.flatten_size,h_dim),linear_block(h_dim,z_dim))
        # self.mu = nn.Linear(z_dim,z_dim)
        # self.logvar = nn.Sequential(nn.Linear(z_dim,z_dim),nn.Hardtanh(min_val=-5.0, max_val=5.0)) # clipping to avoid numerical instabilities

        # Normalise with learnable scale and shift
        self.norm = nn.InstanceNorm1d(self.n_cc, affine=True)

        self.gru = nn.GRU(
            input_size=self.n_cc,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=self.bidirectional
        )

        self.dense = nn.Linear(self.hidden_size * 2 if self.bidirectional else self.hidden_size, self.z_dim)

        # Note this for an AE, try with VAE, for mu and logvar
        self.mu = nn.Linear(z_dim,z_dim)
        self.logvar = nn.Sequential(nn.Linear(z_dim,z_dim),nn.Hardtanh(min_val=-5.0, max_val=5.0)) # clipping to avoid numerical instabilities

        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels=64, sample_rate=44100, n_stft=(1024 // 2 + 1))


    def encode(self, x):

        # Step 1 - Get cepstral coefficients from audio
        # Alternative could be to use torchaudio.transforms.LFCC?
        # transform = torchaudio.transforms.LFCC(
        #     sample_rate=44100,
        #     n_lfcc=128,
        #     speckwargs={"n_fft": self.l_grain, "hop_length": self.hop_size, "center": False},
        # )
        # lfcc = transform(x)

        # This turns our sample into overlapping grains
        mb_grains = F.conv1d(x.unsqueeze(1),self.slice_kernel,stride=self.hop_size,groups=1,bias=None)
        mb_grains = mb_grains.permute(0,2,1)
        bs = mb_grains.shape[0]
        # repeat the overlap add windows acrosss the batch and apply to the grains
        mb_grains = mb_grains*(self.ola_windows.unsqueeze(0).repeat(bs,1,1))
        grain_fft = torch.fft.rfft(mb_grains)
        # use safe log here
        grain_db = 20*safe_log10(torch.abs(grain_fft))
        # grain_db = 20*torch.log10(torch.abs(grain_fft))

        # If taking mel scale
        # grain_fft = grain_fft.permute(0,2,1)
        # grain_mel = safe_log10(self.mel_scale(torch.pow(torch.abs(grain_fft),2)))
        # grain_mel = grain_mel.permute(0,2,1)

        cepstral_coeff = dct.dct(grain_db)
        # Take the first n_cc cepstral coefficients
        cepstral_coeff = cepstral_coeff[:,:,:self.n_cc]

        # Step 2 - Normalize
        # perform permutation to order as [bs, cc, grains], so that s scale and shift it learned for each CC
        cepstral_coeff = cepstral_coeff.permute(0,2,1)
        z = self.norm(cepstral_coeff)
        # permute back to [bs, grains, cc]
        z = z.permute(0, 2, 1)

        # Step 3 - RNN
        z, _ = self.gru(z)

        # Step 4 - dense layer
        z = self.dense(z)

        # Step 5 - mu and log var layers
        mu = self.mu(z)
        logvar = self.logvar(z)
        z = sample_from_distribution(mu, logvar)
 
        # return z
        return z, mu, logvar

    def forward(self, audio):

        # z, mu, logvar = self.encode(audio)
        z= self.encode(audio)

        # return z, mu, logvar
        return z

# Waveform decoder consists simply of dense layers.
class CepstralCoeffsDecoder(nn.Module):

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
                    hop_size,
                    normalize_ola,
                    pp_chans,
                    pp_ker,
                    z_dim,
                    n_mlp_units=512,
                    n_mlp_layers = 3,
                    relu = nn.ReLU,
                    inplace = True,
                    hidden_size = 512,
                    bidirectional = False,
                    n_freq = 1025,
                    l_grain = 2048,
                    hop_ratio = 0.25
                    ):
        super(CepstralCoeffsDecoder, self).__init__()


        self.n_grains = n_grains
        self.l_grain = l_grain
        self.filter_size = l_grain//2+1
        self.tar_l = int((n_grains+3)/4*l_grain)
        self.normalize_ola = normalize_ola
        self.pp_chans = pp_chans
        self.pp_ker = pp_ker
        self.z_dim = z_dim
        self.n_mlp_units = n_mlp_units
        self.n_mlp_layers = n_mlp_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.n_freq = n_freq
        self.relu = relu
        self.inplace = inplace
        self.hop_ratio = hop_ratio

        # self.mlp_z = MLP(n_input=self.z_dim, n_units=self.n_mlp_units, n_layer=self.n_mlp_layers)
        mlp_z = [nn.Sequential(nn.Linear(self.z_dim,self.n_mlp_units),nn.LayerNorm(self.n_mlp_units),self.relu(inplace=self.inplace))]
        mlp_z += [nn.Sequential(nn.Linear(self.n_mlp_units,self.n_mlp_units),nn.LayerNorm(self.n_mlp_units),self.relu(inplace=self.inplace)) for i in range(1, self.n_mlp_layers)]
        self.mlp_z = nn.Sequential(*mlp_z)
        self.n_mlp = 1

        self.gru = nn.GRU(
            input_size = self.n_mlp * self.n_mlp_units,
            hidden_size = self.hidden_size,
            num_layers = 1,
            batch_first = True,
            bidirectional = self.bidirectional,
        )

        # self.mlp_gru = MLP(
        #     n_input=self.hidden_size * 2 if self.bidirectional else self.hidden_size,
        #     n_units=self.n_mlp_units,
        #     n_layer=self.n_mlp_layers,
        #     inplace=True,
        # )
        mlp_gru = [nn.Sequential(nn.Linear((self.hidden_size * 2 if self.bidirectional else self.hidden_size),self.n_mlp_units),nn.LayerNorm(self.n_mlp_units),self.relu(inplace=True))]
        mlp_gru += [nn.Sequential(nn.Linear(self.n_mlp_units,self.n_mlp_units),nn.LayerNorm(self.n_mlp_units),self.relu(inplace=True)) for i in range(1, self.n_mlp_layers)]
        self.mlp_gru = nn.Sequential(*mlp_gru)

        # TODO Try having final dense lapyre map straight to the transfer function for FIR, or try have it predicting CCs
        # NOTE DDSP maps straight to transfer function for FIR fitlers
        self.dense = nn.Linear(self.n_mlp_units, self.n_freq)

        self.filter_window = nn.Parameter(torch.fft.fftshift(torch.hann_window(l_grain)),requires_grad=False).to(DEVICE)

        # Overlap and Add windows for each grain, first half of first grain has no window (ie all values = 1) and 
        # last half of the last grain has no window (ie all values = 1), to allow preserverance of attack and decay.
        ola_window = signal.hann(l_grain,sym=False)
        ola_windows = torch.from_numpy(ola_window).unsqueeze(0).repeat(n_grains,1).type(torch.float32)
        ola_windows[0,:l_grain//2] = ola_window[l_grain//2] # start of 1st grain is not windowed for preserving attacks
        ola_windows[-1,l_grain//2:] = ola_window[l_grain//2] # end of last grain is not wondowed to preserving decays
        self.ola_windows = nn.Parameter(ola_windows,requires_grad=False)

        # Folder
        # Folds input tensor into shape [bs, channels, tar_l, 1], using a kernel size of l_grain, and stride of hop_size
        # can see doc here, https://pytorch.org/docs/stable/generated/torch.nn.Fold.html
        self.ola_folder = nn.Fold((self.tar_l,1),(l_grain,1),stride=(hop_size,1))

        # Normalize OLA
        # This attempts to normalize the energy by dividing by the number of 
        # overlapping grains used when folding to get each point in times energy (amplitude).
        if normalize_ola:
            unfolder = nn.Unfold((l_grain,1),stride=(hop_size,1))
            input_ones = torch.ones(1,1,self.tar_l,1)
            ola_divisor = self.ola_folder(unfolder(input_ones)).squeeze()
            self.ola_divisor = nn.Parameter(ola_divisor,requires_grad=False)
        
        # TODO Look at NGS paper, and ref paper as to how and why this works.
        self.post_pro = nn.Sequential(nn.Conv1d(self.pp_chans, 1, self.pp_ker, padding=pp_ker//2),nn.Softsign())

    def decode(self, z, n_grains=None, ola_windows=None, ola_folder=None, ola_divisor=None):

        # Step 1 - MLP
        h = self.mlp_z(z)

        # Step 2 - RNN
        # note that the GRU (and LSTM) will return a hidden state for each time step, 
        # in theory if trying to predict the next value then you would take the final time step, h[:, -1, :].
        h, _= self.gru(h)

        # Step 3 - MLP
        h = self.mlp_gru(h)

        # Step 4 - Dense Layer
        h = self.dense(h)

        h = mod_sigmoid(h)

        # return the spectral shape 
        # CC part
        # filter_coeffs = h
        # print("Filter shape: ", filter_coeffs.shape)
        # filter_coeffs[:, 128:] = 0.0

        # # NOTE Do i need to  do the scaling back from decibels, also note this introduces 
        # # NOTE is there a torch implementation of this, bit of a bottleneck if not?
        # # NOTE issue with gradient flowwing back
        # # NOTE changed this from idct_2d to idct
        # # inv_filter_coeffs = (dct.idct(filter_coeffs))
        # inv_filter_coeffs = 10**(dct.idct(filter_coeffs) / 20)

        # Step 5 - Noise Filtering
        # Reshape for noise filtering - TODO Look if this is necesary
        h = h.reshape(h.shape[0]*h.shape[1],h.shape[2])

        # Noise filtering (Maybe try using the new noise filtering function and compare to current method...)
        # TODO add hop ratio
        audio = noise_filtering(h, self.filter_window, self.n_grains, self.l_grain, self.hop_ratio)

        # Check if number of grains wanted is entered, else use the original
        if n_grains is None:
            audio = audio.reshape(-1,self.n_grains,self.l_grain)
        else:
            audio = audio.reshape(-1,n_grains,self.l_grain)
        bs = audio.shape[0]

        # Check if an overlapp add window has been passed, if not use that used in encoding.
        if ola_windows is None:
            audio = audio*(self.ola_windows.unsqueeze(0).repeat(bs,1,1))
        else:
            audio = audio*(ola_windows.unsqueeze(0).repeat(bs,1,1))


        # Overlap add folder, folds and reshapes audio into target dimensions, so [bs, tar_l]
        # This is essentially folding and adding the overlapping grains back into original sample size, based on the given hop size.
        # Note that shape is changed here, so that input tensor is of chape [bs, channel X (kernel_size), L],
        # since kernel size is l_grain, this is needed in the second dimension.
        if ola_folder is None:
            audio_sum = self.ola_folder(audio.permute(0,2,1)).squeeze()
        else:
            audio_sum = ola_folder(audio.permute(0,2,1)).squeeze()

        # Normalise the energy values across the audio samples
        if self.normalize_ola:
            if ola_divisor is None:
                # Normalises based on number of overlapping grains used in folding per point in time.
                audio_sum = audio_sum/self.ola_divisor.unsqueeze(0).repeat(bs,1)
            else:
                audio_sum = audio_sum/ola_divisor.unsqueeze(0).repeat(bs,1)

        # NOTE Removed the post processing step for now
        # This module applies a multi-channel temporal convolution that
        # learns a parallel set of time-invariant FIR filters and improves
        # the audio quality of the assembled signal.
        audio_sum = self.post_pro(audio_sum.unsqueeze(1).repeat(1,self.pp_chans,1)).squeeze(1)


        return audio_sum
        # return audio_sum, inv_filter_coeffs.reshape(-1, self.n_grains, inv_filter_coeffs.shape[1]) 

    def forward(self, z, n_grains=None, ola_windows=None, ola_divisor=None):

        audio = self.decode(z, n_grains=n_grains, ola_windows=ola_windows, ola_divisor=ola_divisor)
        # audio = self.decode(z, n_grains=n_grains, ola_windows=ola_windows, ola_divisor=ola_divisor)

        return audio
    
class CepstralCoeffsVAE(nn.Module):

    def __init__(self,
                    n_grains,
                    hop_size,
                    normalize_ola,
                    pp_chans,
                    pp_ker,
                    n_cc=30,
                    hidden_size=512,
                    bidirectional=False,
                    z_dim = 128,
                    l_grain=2048,                    
                    n_mlp_units=512,
                    n_mlp_layers = 3,
                    relu = nn.ReLU,
                    inplace = True,
                    n_freq = 1025,
                    n_linears=3,
                    h_dim=512
                    ):
        super(CepstralCoeffsVAE, self).__init__()

        # Encoder and decoder components
        self.Encoder = CepstralCoeffsEncoder(
                        n_grains = n_grains,
                        hop_size = hop_size,
                        n_cc = n_cc,
                        hidden_size = hidden_size,
                        bidirectional = bidirectional,
                        z_dim = z_dim,
                        l_grain = l_grain,
                    )
        self.Decoder = CepstralCoeffsDecoder(
                        n_grains = n_grains,
                        hop_size = hop_size,
                        normalize_ola = normalize_ola,
                        pp_chans = pp_chans,
                        pp_ker = pp_ker,
                        z_dim = z_dim,
                        n_mlp_units = n_mlp_units,
                        n_mlp_layers = n_mlp_layers,
                        relu = relu,
                        inplace = inplace,
                        hidden_size = hidden_size,
                        bidirectional = bidirectional,
                        n_freq = n_freq,
                        l_grain = l_grain,
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
        
        # z = self.Encoder(x);
        z, mu, log_variance = self.Encoder(x);
        

        # z ---> x_hat
        # Note in paper they also have option passing mu into the decoder and not z
        if sampling:
            x_hat = self.Decoder(z)
        else:
            x_hat, spec = self.Decoder(mu)

        return x_hat, z, mu, log_variance
    
class CepstralCoeffsOnlyEncoder(nn.Module):

    def __init__(self,
                    n_grains,
                    hop_size,
                    n_cc=30,
                    hidden_size=512,
                    bidirectional=False,
                    z_dim = 128,
                    l_grain=2048,
                    ):
        super(CepstralCoeffsOnlyEncoder, self).__init__()


        self.n_grains = n_grains
        self.l_grain = l_grain
        # Set the target length based on hardcodded equation (need to understand why this eq is choosen)
        self.tar_l = int((n_grains+3)/4*l_grain)

        self.n_cc = n_cc
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.z_dim = z_dim
        self.hidden_size = hidden_size

        # define the slice_kernel, this is used in convolution to expand out the grains.
        # TODO look into this a little more, what is eye, identity matrix?
        self.slice_kernel = nn.Parameter(torch.eye(l_grain).unsqueeze(1),requires_grad=False)

        self.hop_size = hop_size

        # Overlap and Add windows for each grain, first half of first grain has no window (ie all values = 1) and 
        # last half of the last grain has no window (ie all values = 1), to allow preserverance of attack and decay.
        ola_window = signal.hann(l_grain,sym=False)
        ola_windows = torch.from_numpy(ola_window).unsqueeze(0).repeat(n_grains,1).type(torch.float32)
        ola_windows[0,:l_grain//2] = ola_window[l_grain//2] # start of 1st grain is not windowed for preserving attacks
        ola_windows[-1,l_grain//2:] = ola_window[l_grain//2] # end of last grain is not wondowed to preserving decays
        self.ola_windows = nn.Parameter(ola_windows,requires_grad=False)


        # self.flatten_size = int(channels*l_grain/(stride**n_convs))
        # self.encoder_linears = nn.Sequential(linear_block(self.flatten_size,h_dim),linear_block(h_dim,z_dim))
        # self.mu = nn.Linear(z_dim,z_dim)
        # self.logvar = nn.Sequential(nn.Linear(z_dim,z_dim),nn.Hardtanh(min_val=-5.0, max_val=5.0)) # clipping to avoid numerical instabilities

        # Normalise with learnable scale and shift
        # self.norm = nn.InstanceNorm1d(self.n_cc, affine=True)
        self.norm = nn.InstanceNorm1d(int(l_grain/2)+1, affine=True)

        self.gru = nn.GRU(
            input_size=int(l_grain/2)+1,
            # input_size=self.n_cc,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=self.bidirectional
        )

        self.dense = nn.Linear(self.hidden_size * 2 if self.bidirectional else self.hidden_size, self.z_dim)

        # Note this for an AE, try with VAE, for mu and logvar
        self.mu = nn.Linear(z_dim,z_dim)
        self.logvar = nn.Sequential(nn.Linear(z_dim,z_dim),nn.Hardtanh(min_val=-5.0, max_val=5.0)) # clipping to avoid numerical instabilities

        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels=64, sample_rate=44100, n_stft=(1024 // 2 + 1))


    def encode(self, x):

        # Step 2 - Normalize
        # perform permutation to order as [bs, cc, grains], so that s scale and shift it learned for each CC
        cepstral_coeff = x.permute(0,2,1)
        z = self.norm(cepstral_coeff)
        # permute back to [bs, grains, cc]
        z = z.permute(0, 2, 1)

        # Step 3 - RNN
        z, _ = self.gru(z)

        # Step 4 - dense layer
        z = self.dense(z)

        # Step 5 - mu and log var layers
        mu = self.mu(z)
        logvar = self.logvar(z)
        z = sample_from_distribution(mu, logvar)
 
        # return z
        return z, mu, logvar

    def forward(self, audio):

        # z, mu, logvar = self.encode(audio)
        z= self.encode(audio)

        # return z, mu, logvar
        return z

# Waveform decoder consists simply of dense layers.
class CepstralCoeffsOnlyDecoder(nn.Module):

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
                    hop_size,
                    normalize_ola,
                    pp_chans,
                    pp_ker,
                    z_dim,
                    n_mlp_units=512,
                    n_mlp_layers = 3,
                    relu = nn.ReLU,
                    inplace = True,
                    hidden_size = 512,
                    bidirectional = False,
                    n_freq = 1025,
                    l_grain = 2048,
                    ):
        super(CepstralCoeffsOnlyDecoder, self).__init__()


        self.n_grains = n_grains
        self.l_grain = l_grain
        self.filter_size = l_grain//2+1
        self.tar_l = int((n_grains+3)/4*l_grain)
        self.normalize_ola = normalize_ola
        self.pp_chans = pp_chans
        self.pp_ker = pp_ker
        self.z_dim = z_dim
        self.n_mlp_units = n_mlp_units
        self.n_mlp_layers = n_mlp_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.n_freq = n_freq
        self.relu = relu
        self.inplace = inplace

        # self.mlp_z = MLP(n_input=self.z_dim, n_units=self.n_mlp_units, n_layer=self.n_mlp_layers)
        mlp_z = [nn.Sequential(nn.Linear(self.z_dim,self.n_mlp_units),nn.LayerNorm(self.n_mlp_units),self.relu(inplace=self.inplace))]
        mlp_z += [nn.Sequential(nn.Linear(self.n_mlp_units,self.n_mlp_units),nn.LayerNorm(self.n_mlp_units),self.relu(inplace=self.inplace)) for i in range(1, self.n_mlp_layers)]
        self.mlp_z = nn.Sequential(*mlp_z)
        self.n_mlp = 1

        self.gru = nn.GRU(
            input_size = self.n_mlp * self.n_mlp_units,
            hidden_size = self.hidden_size,
            num_layers = 1,
            batch_first = True,
            bidirectional = self.bidirectional,
        )

        # self.mlp_gru = MLP(
        #     n_input=self.hidden_size * 2 if self.bidirectional else self.hidden_size,
        #     n_units=self.n_mlp_units,
        #     n_layer=self.n_mlp_layers,
        #     inplace=True,
        # )
        mlp_gru = [nn.Sequential(nn.Linear((self.hidden_size * 2 if self.bidirectional else self.hidden_size),self.n_mlp_units),nn.LayerNorm(self.n_mlp_units),self.relu(inplace=True))]
        mlp_gru += [nn.Sequential(nn.Linear(self.n_mlp_units,self.n_mlp_units),nn.LayerNorm(self.n_mlp_units),self.relu(inplace=True)) for i in range(1, self.n_mlp_layers)]
        self.mlp_gru = nn.Sequential(*mlp_gru)

        # TODO Try having final dense lapyre map straight to the transfer function for FIR, or try have it predicting CCs
        # NOTE DDSP maps straight to transfer function for FIR fitlers
        self.dense = nn.Linear(self.n_mlp_units, self.n_freq)

        self.filter_window = nn.Parameter(torch.fft.fftshift(torch.hann_window(l_grain)),requires_grad=False).to(DEVICE)

        # Overlap and Add windows for each grain, first half of first grain has no window (ie all values = 1) and 
        # last half of the last grain has no window (ie all values = 1), to allow preserverance of attack and decay.
        ola_window = signal.hann(l_grain,sym=False)
        ola_windows = torch.from_numpy(ola_window).unsqueeze(0).repeat(n_grains,1).type(torch.float32)
        ola_windows[0,:l_grain//2] = ola_window[l_grain//2] # start of 1st grain is not windowed for preserving attacks
        ola_windows[-1,l_grain//2:] = ola_window[l_grain//2] # end of last grain is not wondowed to preserving decays
        self.ola_windows = nn.Parameter(ola_windows,requires_grad=False)

        # Folder
        # Folds input tensor into shape [bs, channels, tar_l, 1], using a kernel size of l_grain, and stride of hop_size
        # can see doc here, https://pytorch.org/docs/stable/generated/torch.nn.Fold.html
        self.ola_folder = nn.Fold((self.tar_l,1),(l_grain,1),stride=(hop_size,1))

        # Normalize OLA
        # This attempts to normalize the energy by dividing by the number of 
        # overlapping grains used when folding to get each point in times energy (amplitude).
        if normalize_ola:
            unfolder = nn.Unfold((l_grain,1),stride=(hop_size,1))
            input_ones = torch.ones(1,1,self.tar_l,1)
            ola_divisor = self.ola_folder(unfolder(input_ones)).squeeze()
            self.ola_divisor = nn.Parameter(ola_divisor,requires_grad=False)
        
        # TODO Look at NGS paper, and ref paper as to how and why this works.
        self.post_pro = nn.Sequential(nn.Conv1d(self.pp_chans, 1, self.pp_ker, padding=pp_ker//2),nn.Softsign())

    def decode(self, z, n_grains=None, ola_windows=None, ola_folder=None, ola_divisor=None):

        # Step 1 - MLP
        h = self.mlp_z(z)

        # Step 2 - RNN
        # note that the GRU (and LSTM) will return a hidden state for each time step, 
        # in theory if trying to predict the next value then you would take the final time step, h[:, -1, :].
        h, _= self.gru(h)

        # Step 3 - MLP
        h = self.mlp_gru(h)

        # Step 4 - Dense Layer
        h = self.dense(h)

        h = mod_sigmoid(h)

        # return the spectral shape 
        # CC part
        # filter_coeffs = h
        # print("Filter shape: ", filter_coeffs.shape)
        # filter_coeffs[:, 128:] = 0.0

        # # NOTE Do i need to  do the scaling back from decibels, also note this introduces 
        # # NOTE is there a torch implementation of this, bit of a bottleneck if not?
        # # NOTE issue with gradient flowwing back
        # # NOTE changed this from idct_2d to idct
        # # inv_filter_coeffs = (dct.idct(filter_coeffs))
        # inv_filter_coeffs = 10**(dct.idct(filter_coeffs) / 20)

        return h
        # return audio_sum, inv_filter_coeffs.reshape(-1, self.n_grains, inv_filter_coeffs.shape[1]) 

    def forward(self, z, n_grains=None, ola_windows=None, ola_divisor=None):

        audio = self.decode(z, n_grains=n_grains, ola_windows=ola_windows, ola_divisor=ola_divisor)
        # audio = self.decode(z, n_grains=n_grains, ola_windows=ola_windows, ola_divisor=ola_divisor)

        return audio
    
class CepstralCoeffsOnlyVAE(nn.Module):

    def __init__(self,
                    n_grains,
                    hop_size,
                    normalize_ola,
                    pp_chans,
                    pp_ker,
                    n_cc=30,
                    hidden_size=512,
                    bidirectional=False,
                    z_dim = 128,
                    l_grain=2048,                    
                    n_mlp_units=512,
                    n_mlp_layers = 3,
                    relu = nn.ReLU,
                    inplace = True,
                    n_freq = 1025,
                    n_linears=3,
                    h_dim=512
                    ):
        super(CepstralCoeffsOnlyVAE, self).__init__()

        # Encoder and decoder components
        self.Encoder = CepstralCoeffsOnlyEncoder(
                        n_grains = n_grains,
                        hop_size = hop_size,
                        n_cc = n_cc,
                        hidden_size = hidden_size,
                        bidirectional = bidirectional,
                        z_dim = z_dim,
                        l_grain = l_grain,
                    )
        self.Decoder = CepstralCoeffsOnlyDecoder(
                        n_grains = n_grains,
                        hop_size = hop_size,
                        normalize_ola = normalize_ola,
                        pp_chans = pp_chans,
                        pp_ker = pp_ker,
                        z_dim = z_dim,
                        n_mlp_units = n_mlp_units,
                        n_mlp_layers = n_mlp_layers,
                        relu = relu,
                        inplace = inplace,
                        hidden_size = hidden_size,
                        bidirectional = bidirectional,
                        n_freq = n_freq,
                        l_grain = l_grain,
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
        
        # z = self.Encoder(x);
        z, mu, log_variance = self.Encoder(x);
        

        # z ---> x_hat
        # Note in paper they also have option passing mu into the decoder and not z
        if sampling:
            x_hat = self.Decoder(z)
        else:
            x_hat, spec = self.Decoder(mu)

        return x_hat, z, mu, log_variance
    

class SpectralEncoder(nn.Module):

    def __init__(self,
                    n_grains,
                    hop_size,
                    n_cc=30,
                    hidden_size=512,
                    bidirectional=False,
                    n_mlp_units = 2048,
                    z_dim = 128,
                    l_grain=2048,
                    h_dim = 512
                    ):
        super(SpectralEncoder, self).__init__()


        self.n_grains = n_grains
        self.l_grain = l_grain
        # Set the target length based on hardcodded equation (need to understand why this eq is choosen)
        self.tar_l = int((n_grains+3)/4*l_grain)

        self.n_cc = n_cc
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.z_dim = z_dim
        self.hidden_size = hidden_size
        self.n_mlp_units = n_mlp_units


        # self.dense = nn.Sequential(nn.Linear((self.l_grain//2)+1, self.n_mlp_units), nn.ReLU())
        # # self.dense = nn.Sequential(nn.Linear((self.l_grain//2)+1,self.n_mlp_units), nn.ReLU())
        # self.mu = nn.Linear(self.n_mlp_units, self.z_dim)
        # self.logvar = nn.Sequential(nn.Linear(self.n_mlp_units, self.z_dim),nn.Hardtanh(min_val=-5.0, max_val=5.0)) # clipping to avoid numerical instabilities

        self.flatten_size = int((l_grain//2)+1)
        self.encoder_linears = nn.Sequential(linear_block(self.flatten_size,h_dim),linear_block(h_dim,z_dim))
        self.mu = nn.Linear(z_dim,z_dim)
        self.logvar = nn.Sequential(nn.Linear(z_dim,z_dim),nn.Hardtanh(min_val=-5.0, max_val=5.0)) # clipping to avoid numerical instabilities


    def encode(self, x):

        # z = self.dense(x)

        # mu = self.mu(z)
        # logvar = self.logvar(z)
        # z = sample_from_distribution(mu, logvar)

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

 
        # return z
        return z, mu, logvar

    def forward(self, audio):

        # z, mu, logvar = self.encode(audio)
        z= self.encode(audio)

        # return z, mu, logvar
        return z

# Waveform decoder consists simply of dense layers.
class SpectralDecoder(nn.Module):

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
                    hop_size,
                    normalize_ola,
                    pp_chans,
                    pp_ker,
                    z_dim,
                    n_mlp_units=2048,
                    n_mlp_layers = 3,
                    relu = nn.ReLU,
                    inplace = True,
                    hidden_size = 512,
                    bidirectional = False,
                    n_freq = 1025,
                    l_grain = 2048,
                    n_linears = 1,
                    # n_linears = 3,
                    h_dim = 512
                    ):
        super(SpectralDecoder, self).__init__()


        self.n_grains = n_grains
        self.l_grain = l_grain
        self.filter_size = l_grain//2+1
        self.tar_l = int((n_grains+3)/4*l_grain)
        self.normalize_ola = normalize_ola
        self.pp_chans = pp_chans
        self.pp_ker = pp_ker
        self.z_dim = z_dim
        self.n_mlp_units = n_mlp_units
        self.n_mlp_layers = n_mlp_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.n_freq = n_freq
        self.relu = relu
        self.inplace = inplace



        # self.dense1 = nn.Sequential(nn.Linear(self.z_dim, (self.l_grain//2)+1), nn.ReLU())
        # self.dense2 = nn.Sequential(nn.Linear(self.n_mlp_units, (self.l_grain//2)+1), nn.ReLU())
        # self.dense2 = nn.Sequential(nn.Linear(self.n_mlp_units, self.l_grain, nn.ReLU()))

        decoder_linears = [linear_block(z_dim,h_dim)]
        decoder_linears += [linear_block(h_dim,h_dim) for i in range(1,n_linears)]
        decoder_linears += [nn.Linear(h_dim,self.filter_size)]
        self.decoder_linears = nn.Sequential(*decoder_linears)



    def decode(self, z, n_grains=None, ola_windows=None, ola_folder=None, ola_divisor=None):

        # h = self.dense1(z)
        # filter_coeffs = h

        filter_coeffs = self.decoder_linears(z)

        # What does this do??
        filter_coeffs = mod_sigmoid(filter_coeffs)

        # Reshape back into the batch and grains
        filter_coeffs = filter_coeffs.reshape(-1, self.n_grains, (self.l_grain//2)+1)

        return filter_coeffs
        # return audio_sum, inv_filter_coeffs.reshape(-1, self.n_grains, inv_filter_coeffs.shape[1]) 

    def forward(self, z, n_grains=None, ola_windows=None, ola_divisor=None):

        audio = self.decode(z, n_grains=n_grains, ola_windows=ola_windows, ola_divisor=ola_divisor)
        # audio = self.decode(z, n_grains=n_grains, ola_windows=ola_windows, ola_divisor=ola_divisor)

        return audio
    
class SpectralVAE(nn.Module):

    def __init__(self,
                    n_grains,
                    hop_size,
                    normalize_ola,
                    pp_chans,
                    pp_ker,
                    n_cc=30,
                    hidden_size=512,
                    bidirectional=False,
                    z_dim = 128,
                    l_grain=2048,                    
                    n_mlp_units=512,
                    n_mlp_layers = 3,
                    relu = nn.ReLU,
                    inplace = True,
                    n_freq = 1025,
                    n_linears=1,
                    # n_linears=3,
                    h_dim=512
                    ):
        super(SpectralVAE, self).__init__()

        # Encoder and decoder components
        self.Encoder = SpectralEncoder(
                        n_grains = n_grains,
                        hop_size = hop_size,
                        n_cc = n_cc,
                        hidden_size = hidden_size,
                        bidirectional = bidirectional,
                        z_dim = z_dim,
                        l_grain = l_grain,
                    )
        self.Decoder = SpectralDecoder(
                        n_grains = n_grains,
                        hop_size = hop_size,
                        normalize_ola = normalize_ola,
                        pp_chans = pp_chans,
                        pp_ker = pp_ker,
                        z_dim = z_dim,
                        n_mlp_units = n_mlp_units,
                        n_mlp_layers = n_mlp_layers,
                        relu = relu,
                        inplace = inplace,
                        hidden_size = hidden_size,
                        bidirectional = bidirectional,
                        n_freq = n_freq,
                        l_grain = l_grain,
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
        
        # z = self.Encoder(x);
        z, mu, log_variance = self.Encoder(x);
        

        # z ---> x_hat
        # Note in paper they also have option passing mu into the decoder and not z
        if sampling:
            x_hat = self.Decoder(z)
        else:
            x_hat, spec = self.Decoder(mu)

        return x_hat, z, mu, log_variance

