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

from utils.dsp_components import generate_noise_grains, amp_to_impulse_response, amp_to_impulse_response_w_phase, fft_convolve

torch.manual_seed(0)

train_dataloader,val_dataloader,dataset,tar_l,n_grains,l_grain,hop_size,classes = make_audio_dataloaders(data_dir=AUDIO_DIR,classes=["sea_waves"],sr=SAMPLE_RATE,silent_reject=[0.2,0.2],amplitude_norm=False,batch_size=BATCH_SIZE,hop_ratio=HOP_SIZE_RATIO, tar_l=TARGET_LENGTH,l_grain=GRAIN_LENGTH,high_pass_freq=HIGH_PASS_FREQ,num_workers=0)

for batch, labels in train_dataloader:
    slice_kernel = torch.eye(l_grain).unsqueeze(1)
    mb_grains = F.conv1d(batch.unsqueeze(1), slice_kernel,stride=hop_size,groups=1,bias=None)
    mb_grains = mb_grains.permute(0,2,1)
    bs = mb_grains.shape[0]
    break

audio = mb_grains.reshape(bs*n_grains,l_grain)

# Try MFCC
# mfccs = librosa.feature.mfcc(audio.cpu().numpy(), n_mfcc=10)
# print("MFCCs: ", mfccs.shape)

# Get cepstral coefficients of single grain
print(audio.shape)
grain_fft = fft.rfft(audio.cpu().numpy())
print("Grain Shape:", grain_fft.shape)
# plt.savefig("grain_fft.png")
grain_db = 20*np.log10(np.abs(grain_fft))
inv_grain_fft = torch.from_numpy(fft.ifft(10**(grain_db/20), 2048).real)
print("Inv Grain Shape:", inv_grain_fft.shape)

# ---- MFCC with librosa -----
num_mels = 128
print("Inv Mel STFT: ", grain_fft.shape)
# note i take the squared magnitude below, but do not take sqrt in inversion, due to some normalisaion issue I'm having
mel = librosa.feature.melspectrogram(S=np.abs(grain_fft.T)**2, sr=SAMPLE_RATE, n_mels = num_mels)
print(mel.shape)
inv_mel_grains = librosa.feature.inverse.mel_to_stft(mel, sr=SAMPLE_RATE)
inv_mel_grains = torch.from_numpy(inv_mel_grains.T)
print("Inv Mel STFT: ", inv_mel_grains.shape)

# cepstral_coeff = fft.dct(grain_db)

# # Cut of some of the shape
# cepstral_coeff[:,128:] = 0
# # cepstral_coeff[:,:] = 0

# print("Cepstral Shape: ", cepstral_coeff.shape)

# # cepstral_coeff = fft.dct(grain_fft)
# # plot dct, energy should be mostly in low end, cut out higher end, eyeball the coeff and see where to cut off, maybe around 80hz
# inv_cepstral_coeff_test = (fft.idct(cepstral_coeff))
# inv_cepstral_coeff = 10**(fft.idct(cepstral_coeff) / 20)
# plt.plot(inv_cepstral_coeff_test[0])
# plt.savefig("comparison.png")
# inv_cepstral_coeff = torch.from_numpy(inv_cepstral_coeff)
# print(inv_cepstral_coeff.shape)

# Only take the real part
# inv_cepstral_coeff=torch.view_as_real(inv_cepstral_coeff)[:,0]


# Noise filtering - TODO, try zeroing the phase part
# filter_ir = amp_to_impulse_response(inv_cepstral_coeff, l_grain)
# filter_ir = amp_to_impulse_response_w_phase(inv_cepstral_coeff, l_grain)

# Remove windowing by removing the below function altogether.

print("In Mel grains shape:", inv_mel_grains.shape)
print("In Mel grains shape:", mel.shape)


filter_ir = amp_to_impulse_response_w_phase(inv_mel_grains, l_grain)

noise = generate_noise_grains(bs, n_grains, l_grain, filter_ir.dtype, filter_ir.device, hop_ratio=0.25)
noise = noise.reshape(bs*n_grains, l_grain)

audio = fft_convolve(noise, filter_ir)
audio = audio.reshape(-1,n_grains,l_grain)
# Check if an overlapp add window has been passed, if not use that used in encoding.

ola_window = signal.hann(l_grain,sym=False)
ola_windows = torch.from_numpy(ola_window).unsqueeze(0).repeat(n_grains,1).type(torch.float32)
ola_windows[0,:l_grain//2] = ola_window[l_grain//2] # start of 1st grain is not windowed for preserving attacks
ola_windows[-1,l_grain//2:] = ola_window[l_grain//2] # end of last grain is not wondowed to preserving decays
ola_windows = ola_windows
audio = audio*(ola_windows.unsqueeze(0).repeat(bs,1,1))

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

#for testing
for i in range(audio_sum.shape[0]):
    print(audio_sum[i].shape)
    torchaudio.save(f'{RECONSTRUCTION_SAVE_DIR}/recon_{i}.wav', audio_sum[i].unsqueeze(0).cpu(), SAMPLE_RATE)
    torchaudio.save(f"{RECONSTRUCTION_SAVE_DIR}/{i}.wav", batch[i].unsqueeze(0).cpu(), SAMPLE_RATE)
