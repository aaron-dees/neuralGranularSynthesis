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
class MagSpecEncoder(nn.Module):

    def __init__(self,
                    n_grains,
                    hop_size,
                    n_cc=30,
                    hidden_size=512,
                    bidirectional=False,
                    z_dim = 128,
                    l_grain=2048,
                    ):
        super(MagSpecEncoder, self).__init__()


        self.n_grains = n_grains
        self.l_grain = l_grain
        # Set the target length based on hardcodded equation (need to understand why this eq is choosen)
        self.tar_l = int((n_grains+3)/4*l_grain)
        self.fft_size = (l_grain // 2) + 1
        self.n_cc = n_cc 
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.z_dim = z_dim
        self.hidden_size = hidden_size
        self.hop_size = hop_size
        self.flatten_size = self.fft_size * 861

        # Flatten
        self.flatten = nn.Flatten()

        # Dense Layer
        self.mu = nn.Linear(self.flatten_size,z_dim)
        self.logvar = nn.Linear(self.flatten_size,z_dim)
        # self.logvar = nn.Sequential(nn.Linear(z_dim,z_dim),nn.Hardtanh(min_val=-5.0, max_val=5.0)) # clipping to avoid numerical instabilities

    def encode(self, x):

        # Move all this to the data loasder ??
        # hann_window = torch.hann_window(1024)
        # stft = torch.stft(x, n_fft = 1024,  hop_length=self.hop_size, window=hann_window, return_complex = True)
        # pow_spec = np.abs(stft)**2
        # # Normalsie
        # pow_spec = (pow_spec - pow_spec.min()) / (pow_spec.max() - pow_spec.min())

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
class MagSpecDecoder(nn.Module):

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
                    ):
        super(MagSpecDecoder, self).__init__()


        self.n_grains = n_grains
        self.l_grain = l_grain
        self.filter_size = l_grain//2+1
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

        self.flatten_size = self.filter_size * 861

        self.dense =  nn.Sequential(nn.Linear(self.z_dim, self.flatten_size), nn.Sigmoid())


    def decode(self, z, n_grains=None, ola_windows=None, ola_folder=None, ola_divisor=None):


        # Step 1
        h = self.dense(z)

        # Rehsape
        h = h.reshape(h.shape[0], self.filter_size, 861)

        # transform = torchaudio.transforms.GriffinLim(n_fft = 1024, hop_length = self.hop_size, power=2)
        # audio_recon = transform(h)
        
        return h
        # return audio_sum, inv_filter_coeffs.reshape(-1, self.n_grains, inv_filter_coeffs.shape[1]) 

    def forward(self, z, n_grains=None, ola_windows=None, ola_divisor=None):

        audio = self.decode(z, n_grains=n_grains, ola_windows=ola_windows, ola_divisor=ola_divisor)

        return audio
    
class MagSpecVAE(nn.Module):

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
                    h_dim=512
                    ):
        super(MagSpecVAE, self).__init__()

        # Encoder and decoder components
        self.Encoder = MagSpecEncoder(
                        n_grains = n_grains,
                        hop_size = hop_size,
                        n_cc = n_cc,
                        hidden_size = hidden_size,
                        bidirectional = bidirectional,
                        z_dim = z_dim,
                        l_grain = l_grain,
                    )
        self.Decoder = MagSpecDecoder(
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


###########
# Old Model
###########

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
        # self.norm = nn.InstanceNorm1d(self.n_cc, affine=True)
        self.norm = nn.InstanceNorm1d(513, affine=True)

        self.gru = nn.GRU(
            # input_size=self.n_cc,
            input_size=513,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=self.bidirectional
        )

        self.dense = nn.Linear(self.hidden_size * 2 if self.bidirectional else self.hidden_size, self.z_dim)

        # Note this for an AE, try with VAE, for mu and logvar
        # self.mu = nn.Linear(z_dim,z_dim)
        # self.logvar = nn.Sequential(nn.Linear(z_dim,z_dim),nn.Hardtanh(min_val=-5.0, max_val=5.0)) # clipping to avoid numerical instabilities



    def encode(self, x):

        hann_window = torch.hann_window(1024)
        test_stft = torch.stft(x, n_fft = 1024,  hop_length=self.hop_size, window=hann_window, return_complex = True)
        # print("STFT Shape: ", test_stft.shape)
        # print(torch.abs(test_stft).sum())
        # grain_db = 20*torch.log10(torch.abs(test_stft))
        # print(grain_db.sum())
        # cepstral_coeff = dct.dct(grain_db)
        # cepstral_coeff = torch.from_numpy(cepstral_coeff)
        # print("CC Shape 1: ", cepstral_coeff.shape)
        # cepstral_coeff = cepstral_coeff.permute(0, 2, 1)
        # # cepstral_coeff[:, :, 513:] = 0
        # print("CC Shape 2: ", cepstral_coeff.shape)
        # cepstral_coeff = cepstral_coeff.permute(0, 2, 1)
        # print("CC Shape 3: ", cepstral_coeff.shape)
        # cepstral_coeff = cepstral_coeff.cpu().numpy()
        # print(dct.idct(cepstral_coeff).sum())
        # inv_cepstral_coeff = 10**(dct.idct(cepstral_coeff) / 20)
        # print(inv_cepstral_coeff.sum())
        # transform = torchaudio.transforms.GriffinLim(n_fft = 1024, hop_length = self.hop_size, power=2)
        # recon_audio = transform(torch.abs(test_stft)**2)
        # # recon_audio = torch.from_numpy(librosa.griffinlim(inv_cepstral_coeff, n_fft = 1024, hop_length = self.hop_size))
        # # torchaudio.save(f'recon_audio.wav', recon_audio, 44100)



        # Step 2 - Normalize
        z = self.norm(torch.abs(test_stft)**2)
        # permute to [bs, grains, cc]
        z = z.permute(0, 2, 1)

        # Step 3 - RNN
        z, _ = self.gru(z)


        # Step 4 - dense layer
        z = self.dense(z)

        # Step 5 - mu and log var layers
        # mu = self.mu(z)
        # logvar = self.logvar(z)
        # z = sample_from_distribution(mu, logvar)
 
        return z
        # return z, mu, logvar

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
                    ):
        super(CepstralCoeffsDecoder, self).__init__()


        self.n_grains = n_grains
        self.l_grain = l_grain
        self.filter_size = l_grain//2+1
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
        # self.post_pro = nn.Sequential(nn.Conv1d(pp_chans, 1, pp_ker, padding=pp_ker//2),nn.Softsign())

    def decode(self, z, n_grains=None, ola_windows=None, ola_folder=None, ola_divisor=None):

        # Step 1 - MLP
        h = self.mlp_z(z)

        # Step 2 - RNN
        h, _= self.gru(h)

        # Step 3 - MLP
        h = self.mlp_gru(h)

        # Step 4 - Dense Layer
        h = self.dense(h)

        h = mod_sigmoid(h)

        # reshape int
        h = h.permute(0, 2, 1)

        transform = torchaudio.transforms.GriffinLim(n_fft = 1024, hop_length = self.hop_size, power=2)
        audio_sum = transform(h)

        return audio_sum
        # return audio_sum, inv_filter_coeffs.reshape(-1, self.n_grains, inv_filter_coeffs.shape[1]) 

    def forward(self, z, n_grains=None, ola_windows=None, ola_divisor=None):

        audio = self.decode(z, n_grains=n_grains, ola_windows=ola_windows, ola_divisor=ola_divisor)
        # audio = self.decode(z, n_grains=n_grains, ola_windows=ola_windows, ola_divisor=ola_divisor)

        return audio
    
class CepstralCoeffsAE(nn.Module):

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
                    h_dim=512
                    ):
        super(CepstralCoeffsAE, self).__init__()

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
                    )

        # Number of convolutional layers
    def encode(self, x):

         # x ---> z
        z= self.Encoder(x);
        # z, mu, log_variance = self.Encoder(x);
    
        return {"z":z} 
        # return {"z":z,"mu":mu,"logvar":log_variance} 

    def decode(self, z):
            
        x_hat = self.Decoder(z)
            
        return {"audio":x_hat}

    def forward(self, x, sampling=True):

        # x ---> z
        
        z = self.Encoder(x);
        # z, mu, log_variance = self.Encoder(x);
        

        # z ---> x_hat
        # Note in paper they also have option passing mu into the decoder and not z
        # if sampling:
        x_hat = self.Decoder(z)
        # else:
        #     x_hat, spec = self.Decoder(mu)

        return x_hat, z
