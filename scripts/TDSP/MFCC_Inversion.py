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

from utils.dsp_components import amp_to_impulse_response, amp_to_impulse_response_w_phase, fft_convolve
from utils.utilities import generate_noise_grains

torch.manual_seed(0)

train_dataloader,val_dataloader,dataset,tar_l,n_grains,l_grain,hop_size,classes = make_audio_dataloaders(data_dir=AUDIO_DIR,classes=["sea_waves"],sr=SAMPLE_RATE,silent_reject=[0.2,0.2],amplitude_norm=False,batch_size=BATCH_SIZE,hop_ratio=HOP_SIZE_RATIO, tar_l=TARGET_LENGTH,l_grain=GRAIN_LENGTH,high_pass_freq=HIGH_PASS_FREQ,num_workers=0)

for batch, labels in train_dataloader:
    stft = librosa.stft(batch.squeeze().cpu().numpy(), hop_length=hop_size)
    # Power spec -> Mel Power Spec
    mel = librosa.feature.melspectrogram(S=np.abs(stft)**2, sr=SAMPLE_RATE, n_mels=128)
    # Mel Power Spec -> MFCC
    mfcc = librosa.feature.mfcc(S=mel, n_mfcc=128)
    # MFCC -> Mel Power Spec
    inv_mfcc = librosa.feature.inverse.mfcc_to_mel(mfcc)
    print("Inv MFCC shape:", inv_mfcc.shape)
    # Mel Power Spec -> Linear Mag Spec
    inv_mel = librosa.feature.inverse.mel_to_stft(mel, sr=SAMPLE_RATE)
    print("Inv Mel shape:", inv_mel.shape)
    #Note when using isftf the results are much worse that griffin lim
    recon_audio_orig = librosa.griffinlim(inv_mel, hop_length=hop_size)
    # recon_audio_orig = librosa.istft(inv_mel, hop_length=hop_size)
    # For some reason when using the inv_mel, error start to appear
    # Griffin lim gives and approc
    # recon_audio_orig = librosa.griffinlim(stft, hop_length=hop_size)
    # istft gives a better reconstruction, but gives much worse when using inverted mel.
    # STFT -> Audio
    # recon_audio_orig = librosa.istft(stft, hop_length=hop_size)
    recon_audio_orig = torch.from_numpy(recon_audio_orig).unsqueeze(dim=0)

    print(recon_audio_orig.shape)
    print("Librosa stft: ", stft.shape)
    print("Librosa mel: ", mel.shape)
    print("Librosa inv_mel: ", inv_mel.shape)
    mel = librosa.feature.melspectrogram(y=batch.squeeze().cpu().numpy(), sr=SAMPLE_RATE, hop_length=hop_size)
    recon_audio_lib_rosa_orig = librosa.feature.inverse.mel_to_audio(mel, sr=SAMPLE_RATE, hop_length=hop_size)
    # mfcc = librosa.feature.mfcc(y=batch.squeeze().cpu().numpy(), sr=SAMPLE_RATE, hop_length=hop_size, n_mfcc=128)
    # recon_audio_lib_rosa_orig = librosa.feature.inverse.mfcc_to_audio(mfcc, sr=SAMPLE_RATE, hop_length=hop_size)

    recon_audio_lib_rosa_orig = torch.from_numpy(recon_audio_lib_rosa_orig).unsqueeze(dim=0)
    break

# Save the reconstructed mel
for i in range(recon_audio_orig.shape[0]):
    torchaudio.save(f'{RECONSTRUCTION_SAVE_DIR}/recon_mel_{i}.wav', recon_audio_orig[i].unsqueeze(0).cpu(), SAMPLE_RATE)
    torchaudio.save(f'{RECONSTRUCTION_SAVE_DIR}/recon_mel_librosa_{i}.wav', recon_audio_lib_rosa_orig[i].unsqueeze(0).cpu(), SAMPLE_RATE)