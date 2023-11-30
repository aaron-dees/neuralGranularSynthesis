import sys
sys.path.append('../../')

import torch
import torchaudio
from torch.nn import functional as F
from models.dataloaders.waveform_dataloaders import ESC50WaveformDataset, make_audio_dataloaders
from scripts.configs.hyper_parameters_waveform import *
from scipy import signal, fft
import matplotlib.pyplot as plt
import numpy as np
import librosa

import utils.dsp_components as dsp
import utils.utilities as utils

# torch.manual_seed(0)

# Torch dataloader bits
train_dataloader,val_dataloader,dataset,tar_l,n_grains,l_grain,hop_size,classes = make_audio_dataloaders(data_dir=AUDIO_DIR,classes=["sea_waves"],sr=SAMPLE_RATE,silent_reject=[0.2,0.2],amplitude_norm=False,batch_size=BATCH_SIZE,hop_ratio=HOP_SIZE_RATIO, tar_l=TARGET_LENGTH,l_grain=GRAIN_LENGTH,high_pass_freq=HIGH_PASS_FREQ,num_workers=0)
test_set = torch.utils.data.Subset(dataset, range(0,TEST_SIZE))
test_dataloader = torch.utils.data.DataLoader(test_set, batch_size = TEST_SIZE, shuffle=False, num_workers=0)

# Get grains and add windowing
for batch, labels in test_dataloader:
    slice_kernel = torch.eye(l_grain).unsqueeze(1)
    mb_grains = F.conv1d(batch.unsqueeze(1), slice_kernel,stride=hop_size,groups=1,bias=None)
    mb_grains = mb_grains.permute(0,2,1)
    bs = mb_grains.shape[0]
    print(mb_grains.shape)
    # Add windowing
    ola_window = signal.hann(l_grain,sym=False)
    ola_windows = torch.from_numpy(ola_window).unsqueeze(0).repeat(n_grains,1).type(torch.float32)
    ola_windows[0,:l_grain//2] = ola_window[l_grain//2] # start of 1st grain is not windowed for preserving attacks
    ola_windows[-1,l_grain//2:] = ola_window[l_grain//2] # end of last grain is not wondowed to preserving decays
    ola_windows = torch.nn.Parameter(ola_windows,requires_grad=False)
    mb_grains = mb_grains*(ola_windows.unsqueeze(0).repeat(bs,1,1))
    break

mb_grains = mb_grains.reshape(bs*n_grains,l_grain)


# Get cepstral coefficients for each grain
grain_fft = fft.rfft(mb_grains.cpu().numpy())
grain_db = 20*np.log10(np.abs(grain_fft))
cepstral_coeff = fft.dct(grain_db)
# Cut of some of the shape
cepstral_coeff[:,128:] = 0

# Invert the cepstral coefficitents and scale from db -> power scale.
inv_cepstral_coeff = 10**(fft.idct(cepstral_coeff) / 20)
inv_cepstral_coeff = torch.from_numpy(inv_cepstral_coeff)

# Get the impulse response from the inverted cepstral coefficients
filter_ir = dsp.amp_to_impulse_response_w_phase(inv_cepstral_coeff, l_grain)

# Generate the noise grains
noise = utils.generate_noise_grains(bs, n_grains, l_grain, filter_ir.dtype, filter_ir.device, hop_ratio=0.25)
noise = noise.reshape(bs*n_grains, l_grain)

# Convolve the noise and impulse response 
audio = dsp.fft_convolve(noise, filter_ir)
audio = audio.reshape(-1,n_grains,l_grain)

# Apply same window as previously to new filtered noise grains
ola_window = signal.hann(l_grain,sym=False)
ola_windows = torch.from_numpy(ola_window).unsqueeze(0).repeat(n_grains,1).type(torch.float32)
ola_windows[0,:l_grain//2] = ola_window[l_grain//2] # start of 1st grain is not windowed for preserving attacks
ola_windows[-1,l_grain//2:] = ola_window[l_grain//2] # end of last grain is not wondowed to preserving decays
ola_windows = ola_windows
audio = audio*(ola_windows.unsqueeze(0).repeat(bs,1,1))

# Apply simply normalisation
audio = audio/torch.max(audio)


# Overlap add folder, folds and reshapes audio into target dimensions, so [bs, tar_l]
# This is essentially folding and adding the overlapping grains back into original sample size, based on the given hop size.
# Note that shape is changed here, so that input tensor is of chape [bs, channel X (kernel_size), L],
# since kernel size is l_grain, this is needed in the second dimension.
ola_folder = torch.nn.Fold((tar_l,1),(l_grain,1),stride=(hop_size,1))
audio_sum = ola_folder(audio.permute(0,2,1)).squeeze()


# Normalise the energy values across the audio samples
if NORMALIZE_OLA:
    # Normalises based on number of overlapping grains used in folding per point in time.
    unfolder = torch.nn.Unfold((l_grain,1),stride=(hop_size,1))
    input_ones = torch.ones(1,1,tar_l,1)
    ola_divisor = ola_folder(unfolder(input_ones)).squeeze()
    ola_divisor = ola_divisor
    audio_sum = audio_sum/ola_divisor.unsqueeze(0).repeat(bs,1)
else:
    audio_sum = audio_sum.unsqueeze(0)

#for testing
for i in range(audio_sum.shape[0]):
    print(audio_sum[i].shape)
    torchaudio.save(f'{RECONSTRUCTION_SAVE_DIR}/recon_{i}.wav', audio_sum[i].unsqueeze(0).cpu(), SAMPLE_RATE)
    torchaudio.save(f"{RECONSTRUCTION_SAVE_DIR}/{i}.wav", batch[i].unsqueeze(0).cpu(), SAMPLE_RATE)
