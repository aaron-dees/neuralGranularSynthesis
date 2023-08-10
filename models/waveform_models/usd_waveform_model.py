import sys
sys.path.append('../')

from utils.utilities import sample_from_distribution
from scripts.configs.hyper_parameters_waveform import BATCH_SIZE, DEVICE, LATENT_SIZE
from utils.dsp_components import noise_filtering, mod_sigmoid


import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.nn import functional as F
import numpy as np
from scipy import signal

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

        # define the slice_kernel, what is this?
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
                    z_dim=128
                    ):
        super(WaveformDecoder, self).__init__()

        self.n_grains = n_grains
        self.l_grain = l_grain
        self.filter_size = l_grain//2+1
        self.tar_l = int((n_grains+3)/4*l_grain)
        self.normalize_ola = normalize_ola
        self.pp_chans = pp_chans

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

        audio = noise_filtering(filter_coeffs, self.filter_window)

        # use below if we were using grains
        # audio = audio.reshape(-1,n_grains,self.hparams.l_grain)
        audio = audio.reshape(-1,1,self.l_grain)


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

    def decode(self, z, mu, sampling=True):

        if sampling:
            x_hat = self.Decoder(z)
        else:
            x_hat = self.Decoder(mu) 
        
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