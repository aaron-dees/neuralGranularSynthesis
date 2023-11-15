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

    log_pow_stft = 20*np.log10(np.abs(stft))

    # Note transposing for librosa
    cepstral_coeff = fft.dct(log_pow_stft.T)

    print("CC Shape: ", cepstral_coeff.shape)

    # cepstral_coeff[:, 128:] = 0

    inv_cepstral_coeff = fft.idct(cepstral_coeff)
    inv_cepstral_coeff = 10**(inv_cepstral_coeff/20)
    # Transpose for librosa
    inv_cepstral_coeff = inv_cepstral_coeff.T

    #Note when using isftf the results are much worse that griffin lim
    recon_audio_orig = librosa.griffinlim(inv_cepstral_coeff, hop_length=hop_size)
    # recon_audio_orig = librosa.griffinlim(filtered_noise_fft.cpu().numpy(), hop_length=hop_size)
    recon_audio_orig = recon_audio_orig/np.max(recon_audio_orig)
    recon_audio_orig = torch.from_numpy(recon_audio_orig).unsqueeze(dim=0)

    break

# Save the reconstructed mel
for i in range(recon_audio_orig.shape[0]):
    torchaudio.save(f'{RECONSTRUCTION_SAVE_DIR}/CC_Inversion_{i}.wav', recon_audio_orig[i].unsqueeze(0).cpu(), SAMPLE_RATE)