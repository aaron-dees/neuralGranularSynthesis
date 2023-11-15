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
    stft = librosa.stft(batch.squeeze().cpu().numpy(), hop_length=hop_size)
    # Power spec -> Mel Power Spec
    mel = librosa.feature.melspectrogram(S=np.abs(stft)**2, sr=SAMPLE_RATE)
    # mel = librosa.feature.melspectrogram(S=np.abs(stft)**2, sr=SAMPLE_RATE, n_mels=64)
    # Mel Power Spec -> MFCC
    mfcc = librosa.feature.mfcc(S=mel, n_mfcc=40)
    # mfcc_direct = librosa.feature.mfcc(y=batch.squeeze().cpu().numpy(), hop_length=hop_size)
    # MFCC -> Mel Power Spec
    # inv_mfcc = librosa.feature.inverse.mfcc_to_mel(mfcc, n_mels=64)
    inv_mfcc = librosa.feature.inverse.mfcc_to_mel(mfcc)
    print("Inv MFCC shape:", inv_mfcc.shape)
    # Mel Power Spec -> Linear Mag Spec
    inv_mel = librosa.feature.inverse.mel_to_stft(mel, sr=SAMPLE_RATE)
    print("Inv Mel shape:", inv_mel.shape)

    # Noise filtering
    filter_fft = inv_mel
    print("Filter Shape: ", filter_fft.shape)

    # Time
    noise_stft = generate_noise_grains_stft(batch.shape[0], batch.shape[1], torch.float32, batch.device, hop_size)
    noise_stft = noise_stft.reshape(batch.shape[0]*noise_stft.shape[1], noise_stft.shape[2])

    # NOTE no padding done here, how to I pad correctly in freq domain?
    # Convolve the noise and the filters (in time domain, so multiply in freq)
    filtered_noise_fft = noise_stft * filter_fft

    #Note when using isftf the results are much worse that griffin lim
    recon_audio_orig = librosa.griffinlim(filtered_noise_fft.cpu().numpy(), hop_length=hop_size)
    recon_audio_orig = recon_audio_orig/np.max(recon_audio_orig)
    recon_audio_orig = torch.from_numpy(recon_audio_orig).unsqueeze(dim=0)

    break

# Save the reconstructed mel
for i in range(recon_audio_orig.shape[0]):
    torchaudio.save(f'{RECONSTRUCTION_SAVE_DIR}/recon_mel_{i}.wav', recon_audio_orig[i].unsqueeze(0).cpu(), SAMPLE_RATE)