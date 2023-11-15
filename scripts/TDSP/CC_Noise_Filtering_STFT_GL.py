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

from utils.dsp_components import amp_to_impulse_response, amp_to_impulse_response_w_phase, fft_convolve, fft_convolve_2
from utils.utilities import generate_noise_grains, generate_noise_grains_stft

torch.manual_seed(0)

train_dataloader,val_dataloader,dataset,tar_l,n_grains,l_grain,hop_size,classes = make_audio_dataloaders(data_dir=AUDIO_DIR,classes=["sea_waves"],sr=SAMPLE_RATE,silent_reject=[0.2,0.2],amplitude_norm=False,batch_size=BATCH_SIZE,hop_ratio=HOP_SIZE_RATIO, tar_l=TARGET_LENGTH,l_grain=GRAIN_LENGTH,high_pass_freq=HIGH_PASS_FREQ,num_workers=0)

for batch, labels in train_dataloader:

    # try and pad the windows

    stft = librosa.stft(batch.squeeze().cpu().numpy(), hop_length=hop_size)
    print(stft.shape)

    log_pow_stft = 20*np.log10(np.abs(stft))

    # Note transposing for librosa
    cepstral_coeff = fft.dct(log_pow_stft.T)

    print("CC Shape: ", cepstral_coeff.shape)

    # cepstral_coeff[:, 128] = 0

    inv_cepstral_coeff = fft.idct(cepstral_coeff)
    inv_cepstral_coeff = 10**(inv_cepstral_coeff/20)
    # Transpose for librosa
    inv_cepstral_coeff = inv_cepstral_coeff.T

    # Noise filtering
    filter_fft = inv_cepstral_coeff
    print("Filter Shape: ", filter_fft.shape)

    # Time
    noise_stft = generate_noise_grains_stft(batch.shape[0], batch.shape[1], torch.float32, batch.device, hop_size)
    print("Noise Shape: ", noise_stft.shape)
    noise_stft = noise_stft.reshape(batch.shape[0]*noise_stft.shape[1], noise_stft.shape[2])

    print("Noise Shape: ", noise_stft.shape)

    # NOTE How can I pad the filter??
    # Convolve the noise and the filters (in time domain, so multiply in freq)
    # NOTE I guess I'm throwing away the noises phase here ??
    # padded_noise = fft.irfft(noise_stft.T.cpu().numpy())
    # padded_noise = torch.from_numpy(padded_noise)
    # padded_noise = torch.nn.functional.pad(padded_noise, (0, padded_noise.shape[-1])).cpu().numpy()
    # padded_noise_fft = fft.rfft(padded_noise)

    # padded_filter = fft.irfft(filter_fft.T)
    # padded_filter = torch.from_numpy(padded_filter)
    # padded_filter = torch.nn.functional.pad(padded_filter, (padded_filter.shape[-1], 0)).cpu().numpy()
    # padded_filter_fft = fft.rfft(padded_filter)

    # signal = torch.nn.functional.pad(signal, (0, signal.shape[-1]))
    # kernel = torch.nn.functional.pad(kernel, (kernel.shape[-1], 0))

    filtered_noise_fft = noise_stft * filter_fft
    # Padding example below 
    # filtered_noise_fft = padded_noise_fft.T * padded_filter_fft.T
    # filtered_noise_fft = filtered_noise_fft[filtered_noise_fft.shape[-1] // 2:, ...]


    #Note when using isftf the results are much worse that griffin lim
    # recon_audio_orig = librosa.griffinlim(filtered_noise_fft, hop_length=hop_size)
    print(filtered_noise_fft.shape)
    recon_audio_orig = librosa.griffinlim(filtered_noise_fft.cpu().numpy(), hop_length=hop_size)
    recon_audio_orig = recon_audio_orig/np.max(recon_audio_orig)
    recon_audio_orig = torch.from_numpy(recon_audio_orig).unsqueeze(dim=0)

    break

# Save the reconstructed mel
for i in range(recon_audio_orig.shape[0]):
    torchaudio.save(f'{RECONSTRUCTION_SAVE_DIR}/recon_mel_{i}.wav', recon_audio_orig[i].unsqueeze(0).cpu(), SAMPLE_RATE)