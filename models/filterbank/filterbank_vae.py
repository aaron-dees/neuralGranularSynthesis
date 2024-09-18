import sys
sys.path.append('../')

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

from utils.utilities import sample_from_distribution, generate_noise_grains
from utils.dsp_components import noise_filtering, mod_sigmoid, safe_log10
from models.filterbank.filterbank import FilterBank

def compute_magnitude_filters(filters):
    magnitude_filters = torch.fft.rfft(filters)
    magnitude_filters = torch.abs(magnitude_filters)
    return magnitude_filters

def check_power_of_2(x):
    return 2 ** int(math.log(x, 2)) == x

def get_next_power_of_2(x):
    return int(math.pow(2, math.ceil(math.log(x)/math.log(2))))

def pad_filters(filters, n_samples):
    for i in range(len(filters)):
        filters[i] = np.pad(filters[i], (n_samples-len(filters[i]),0))
    return torch.from_numpy(np.array(filters))

# Builds loopable noise bands, based on filter bank and white nose, see 'deterministic loopable noise bands in original paper'
def get_noise_bands(fb, min_noise_len, normalize):
    #build deterministic loopable noise bands
    if fb.max_filter_len > min_noise_len:
        noise_len = get_next_power_of_2(fb.max_filter_len)
    else:
        noise_len = min_noise_len
    filters = pad_filters(fb.filters, noise_len)
    magnitude_filters = compute_magnitude_filters(filters=filters)
    torch.manual_seed(42) #enforce deterministic noise
    phase_noise = torch.FloatTensor(magnitude_filters.shape[0], magnitude_filters.shape[-1]).uniform_(-math.pi, math.pi).to(magnitude_filters.device)
    phase_noise = torch.exp(1j*phase_noise)
    phase_noise[:,0] = 0
    phase_noise[:,-1] = 0
    magphase = magnitude_filters*phase_noise
    noise_bands = torch.fft.irfft(magphase)
    print(noise_bands.shape)
    print(noise_bands.dtype)
    if normalize:
        noise_bands = (noise_bands / torch.max(noise_bands.abs())) 

    # test noise look by concatonating along x axis.
    # cat = torch.concat((noise_bands[10], noise_bands[10]))
    # cat = torch.concat((cat, noise_bands[10]))
    # print(cat.shape) 
    # torchaudio.save(f'/Users/adees/Code/neural_granular_synthesis/scripts/recon_audio/loopNoise.wav', noise_bands[10].to(torch.float32).unsqueeze(0).cpu(), 44100)
    # torchaudio.save(f'/Users/adees/Code/neural_granular_synthesis/scripts/recon_audio/loopNoiseCat.wav', cat.to(torch.float32).unsqueeze(0).cpu(), 44100)

    return noise_bands.unsqueeze(0).float(), noise_len

# MLP and GRU for v2
def mlp(in_size, hidden_size, n_layers):
    channels = [in_size] + (n_layers) * [hidden_size]
    net = []
    for i in range(n_layers):
        net.append(nn.Linear(channels[i], channels[i + 1]))
        net.append(nn.LayerNorm(channels[i + 1]))
        net.append(nn.LeakyReLU())
    return nn.Sequential(*net)

def gru(n_input, hidden_size):
    return nn.GRU(n_input * hidden_size, hidden_size, batch_first=True)

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

        self.norm = nn.LayerNorm((self.n_grains, self.flatten_size))
        self.gru = nn.GRU(input_size=self.flatten_size, hidden_size=self.h_dim, batch_first=True)
        self.linear = nn.Linear(self.h_dim, self.z_dim)
        self.mu = nn.Linear(self.h_dim, self.z_dim)
        self.logvar = nn.Linear(self.h_dim, self.z_dim)


    def encode(self, x):

        # h = x.reshape(x.shape[0]*self.n_grains,(self.l_grain//2)+1)
        # mfccs = spectral_ops.compute_mfcc(
        #     audio,
        #     lo_hz=20.0,
        #     hi_hz=8000.0,
        #     fft_size=self.fft_size,
        #     mel_bins=128,
        #     mfcc_bins=30,
        #     overlap=self.overlap,
        #     pad_end=True)

        # # Normalize.
        # z = self.z_norm(mfccs[:, :, tf.newaxis, :])[:, :, 0, :]
        # # Run an RNN over the latents.
        # z = self.rnn(z)
        # # Bounce down to compressed z dimensions.
        # z = self.dense_out(z)

        h = self.norm(x)
        # h = h.unsqueeze(-2)
        # z = z.permute(2, 0, 1)
        h = self.gru(h)[0]
        # print(img)
        h = h.reshape(h.shape[0]*self.n_grains,self.h_dim)
        # z = self.linear(z)
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
                    h_dim = 512,
                    n_band = 2048
                    ):
        super(SpectralDecoder_v1, self).__init__()

        self.n_grains = n_grains
        self.l_grain = l_grain
        # self.filter_size = l_grain//2+1
        self.filter_size = n_band
        self.z_dim = z_dim
        self.h_dim = h_dim

        decoder_linears = [linear_block(self.z_dim,self.h_dim)]
        # decoder_linears += [linear_block(h_dim,h_dim) for i in range(1,n_linears)]
        decoder_linears += [nn.Linear(self.h_dim, self.filter_size)]
        self.decoder_linears = nn.Sequential(*decoder_linears)
        self.sigmoid = nn.Sigmoid()

    def decode(self, z, n_grains=None, ola_windows=None, ola_folder=None, ola_divisor=None):

        filter_coeffs = self.decoder_linears(z)

        # What does this do??
        filter_coeffs = mod_sigmoid(filter_coeffs)
        # filter_coeffs = self.sigmoid(filter_coeffs)

        # Reshape back into the batch and grains
        filter_coeffs = filter_coeffs.reshape(-1, self.n_grains, self.filter_size)
        filter_coeffs = filter_coeffs.permute(0,2,1)

        return filter_coeffs

    def forward(self, z, n_grains=None, ola_windows=None, ola_divisor=None):

        audio = self.decode(z, n_grains=n_grains, ola_windows=ola_windows, ola_divisor=ola_divisor)

        return audio


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
                    h_dim = 512,
                    n_band = 2048
                    ):
        super(SpectralDecoder_v2, self).__init__()

        self.n_grains = n_grains
        self.l_grain = l_grain
        # self.filter_size = l_grain//2+1
        self.filter_size = n_band
        self.z_dim = z_dim
        self.h_dim = h_dim

        # original, trating each element in the latent as a control parameter
        # in_mlps = []
        # for i in range(n_control_params):
        # for i in range(self.z_dim):
        #     in_mlps.append(mlp(1, self.h_dim, 1))
        # self.in_mlps = nn.ModuleList(in_mlps)
        # self.gru = gru(self.z_dim, self.h_dim)

        self.in_mlps = mlp(self.z_dim, self.h_dim, 1)
        self.gru = gru(1, self.h_dim)
        self.out_mlp = mlp(self.h_dim + self.z_dim, self.h_dim, 3)
        self.amplitudes_layer = nn.Linear(self.h_dim, n_band)

    def decode(self, z, n_grains=None, ola_windows=None, ola_folder=None, ola_divisor=None):

        hidden = []
        z = z.reshape(-1, self.n_grains, z.shape[1])
        # for i in range(len(self.in_mlps)):
        #     hidden.append(self.in_mlps[i](z[:,:,i].unsqueeze(-1)))
        hidden.append(self.in_mlps(z))
        hidden = torch.cat(hidden, dim=-1)
        hidden = self.gru(hidden)[0]
        # Why do we use the below??
        # for i in range(self.z_dim):
            # hidden = torch.cat([hidden, z[:,:,i].unsqueeze(-1)], dim=-1)
        hidden = torch.cat([hidden, z], dim=-1)
        hidden = self.out_mlp(hidden)
        amplitudes = self.amplitudes_layer(hidden).permute(0,2,1)
        amplitudes = mod_sigmoid(amplitudes)

        # print(z.shape)
        # filter_coeffs = self.decoder_linears(z)

        # # What does this do??
        # filter_coeffs = mod_sigmoid(filter_coeffs)
        # # filter_coeffs = self.sigmoid(filter_coeffs)

        # # Reshape back into the batch and grains
        # filter_coeffs = filter_coeffs.reshape(-1, self.n_grains, self.filter_size)
        # filter_coeffs = filter_coeffs.permute(0,2,1)
        # print(filter_coeffs.shape)

        return amplitudes 

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
                    n_band = 2048,
                    linear_min_f = 20, 
                    linear_max_f_cutoff_fs = 4,
                    fs = 44100,
                    filterbank_attenuation=50,
                    min_noise_len = 2**16,
                    normalize_noise_bands=True,
                    synth_window=32
                    ):
        super(SpectralVAE_v1, self).__init__()

        self.z_dim = z_dim
        self.n_grains = n_grains
        self.l_grain = l_grain
        self.synth_window = synth_window

        fb  = FilterBank(n_filters_linear = n_band//2, n_filters_log = n_band//2, linear_min_f = linear_min_f, linear_max_f_cutoff_fs = linear_max_f_cutoff_fs,  fs = fs, attenuation = filterbank_attenuation)
        print(len(fb.filters))
        self.center_frequencies = fb.band_centers #store center frequencies for reference
        self.noise_bands, self.noise_len = get_noise_bands(fb=fb, min_noise_len=min_noise_len, normalize=normalize_noise_bands)

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
                        n_band=n_band,

                    )

    def synth_batch(self, amplitudes):
        """Apply the predicted amplitudes to the noise bands.
        Args:
        ----------
        amplitudes : torch.Tensor
            Predicted amplitudes

        Returns:
        ----------  
        signal : torch.Tensor
            Output audio signal
        """

        #synth in noise_len frames to fit longer sequences on GPU memory
        frame_len = int(self.noise_len/self.synth_window)
        n_frames = math.ceil(amplitudes.shape[-1]/frame_len)
        self.noise_bands = self.noise_bands.to(amplitudes.device)
        #avoid overfitting to noise values
        self.noise_bands = torch.roll(self.noise_bands, shifts=int(torch.randint(low=0, high=self.noise_bands.shape[-1], size=(1,))), dims=-1)
        signal_len = amplitudes.shape[-1]*self.synth_window
        #smaller amp len than noise_len
        if amplitudes.shape[-1]/frame_len < 1:
            upsampled_amplitudes = F.interpolate(amplitudes, scale_factor=self.synth_window, mode='linear')
            signal = (self.noise_bands[..., :signal_len]*upsampled_amplitudes).sum(1, keepdim=True)
        else:
            for i in range(n_frames):
                if i == 0:
                    upsampled_amplitudes = F.interpolate(amplitudes[..., :frame_len], scale_factor=self.synth_window, mode='linear')
                    signal = (self.noise_bands*upsampled_amplitudes).sum(1, keepdim=True)
                #last iteration
                elif i == (n_frames-1):
                    upsampled_amplitudes = F.interpolate(amplitudes[..., i*frame_len:], scale_factor=self.synth_window, mode='linear')
                    signal = torch.cat([signal, (self.noise_bands[...,:upsampled_amplitudes.shape[-1]]*upsampled_amplitudes).sum(1, keepdim=True)], dim=-1)
                else:
                    upsampled_amplitudes = F.interpolate(amplitudes[..., i*frame_len:(i+1)*frame_len], scale_factor=self.synth_window, mode='linear')
                    signal = torch.cat([signal, (self.noise_bands*upsampled_amplitudes).sum(1, keepdim=True)], dim=-1)
        return signal 

        # Number of convolutional layers
    def encode(self, x):

        # x ---> z
        z, mu, log_variance = self.Encoder(x);
    
        return {"z":z, "mu":mu, "logvar":log_variance} 

    def decode(self, z):
            
        amplitudes = self.Decoder(z)
        signal = self.synth_batch(amplitudes=amplitudes)
            
        return {"audio":signal}

    def forward(self, x, sampling=True):

        # x ---> z
        
        z, mu, log_variance = self.Encoder(x);

        # z ---> x_hat
        # Note in paper they also have option passing mu into the decoder and not z
        if sampling:
            amplitudes = self.Decoder(z)
            signal = self.synth_batch(amplitudes=amplitudes)
        else:
            amplitudes = self.Decoder(mu)
            signal = self.synth_batch(amplitudes=amplitudes)

        return signal, z, mu, log_variance