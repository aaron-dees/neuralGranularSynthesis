import sys
sys.path.append('../../')

import torch
import torchaudio
from torch.nn import functional as F
from models.dataloaders.waveform_dataloaders import ESC50WaveformDataset, make_audio_dataloaders
from scripts.configs.hyper_parameters_inversionExp import *
from scipy import signal, fft
import matplotlib.pyplot as plt
import numpy as np
import librosa
import torch_dct as dct
from models.loss_functions import spectral_distances

import utils.dsp_components as dsp
import utils.utilities as utils

from frechet_audio_distance import FrechetAudioDistance

# to use `vggish`
frechet = FrechetAudioDistance(
    model_name="vggish",
    sample_rate=16000,
    use_pca=False, 
    use_activation=False,
    verbose=False
)

# torch.manual_seed(0)

# Torch dataloader bits
train_dataloader,val_dataloader,dataset,tar_l,n_grains,l_grain,hop_size,classes = make_audio_dataloaders(data_dir=AUDIO_DIR,classes=["sea_waves"],sr=SAMPLE_RATE,silent_reject=[0,0],amplitude_norm=True,batch_size=BATCH_SIZE,hop_ratio=HOP_SIZE_RATIO, tar_l=TARGET_LENGTH,l_grain=GRAIN_LENGTH,high_pass_freq=HIGH_PASS_FREQ,num_workers=0)
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

# TODO Look at the windowed noise grains
# plt.plot(mb_grains[7])
# plt.savefig("test.png")

# Get cepstral coefficients for each grain
# grain_fft = torch.fft.rfft(mb_grains.cpu().numpy())

# ----- CC Inversion -----
grain_fft = torch.fft.rfft(mb_grains)
# grain_db = 20*dsp.safe_log10(np.abs(grain_fft))
grain_db = 20*dsp.safe_log10(np.abs(grain_fft))
cepstral_coeff = dct.dct(grain_db)
# Cut of some of the shape
cepstral_coeff[:,NUM_CC:] = 0
print("Cep Size: ", cepstral_coeff.shape)
# Invert the cepstral coefficitents and scale from db -> power scale.
inv_cepstral_coeff = 10**(dct.idct(cepstral_coeff) / 20)
print("Inverted Size: ", inv_cepstral_coeff.shape)


# ----- MFCC Invesion -----

# Torch functions for inversion
# mel_scale = torchaudio.transforms.MelScale(
#             n_mels=64, sample_rate=44100, n_stft=(1024 // 2 + 1))
# inv_mel_scale = torchaudio.transforms.InverseMelScale(n_stft=(1024 // 2 + 1), n_mels=64, sample_rate=44100)

grain_fft_perm = grain_fft.permute(1,0)
# grain_mel_torch = torch.log10(mel_scale(torch.pow(torch.abs(grain_fft_perm),2)))
# grain_mel= dsp.safe_log(torch.from_numpy((librosa.feature.melspectrogram(S=np.abs(grain_fft_perm)**2, sr=SAMPLE_RATE, n_fft=GRAIN_LENGTH, n_mels=NUM_MELS))))
grain_mel= dsp.safe_log10(torch.from_numpy((librosa.feature.melspectrogram(S=np.abs(grain_fft_perm)**2, sr=SAMPLE_RATE, n_fft=GRAIN_LENGTH, n_mels=NUM_MELS))))
mfccs = dct.dct(grain_mel)
inv_mfccs = dct.idct(mfccs).cpu().numpy()
inv_mfccs = librosa.feature.inverse.mel_to_stft(M=10**inv_mfccs, sr=SAMPLE_RATE, n_fft=GRAIN_LENGTH)
inv_mfccs = torch.from_numpy(inv_mfccs).permute(1,0)
# inv_mel_torch = inv_mel_scale(inv_mel)
# Add a tiny amount to avoid division by zero
# inv_mfccs = inv_mfccs + 1e-8

# Divide the original signal by the inverted CC's
sig_noise_fft_cc = grain_fft / inv_cepstral_coeff
sig_noise_fft_mfcc = grain_fft / inv_mfccs 
# print(sig_noise_fft_cc)
# print(sig_noise_fft_mfcc)
# print(sig_noise_fft_cc.abs().max())
# print(sig_noise_fft_mfcc.abs().max())

sig_noise_cc = torch.fft.irfft(sig_noise_fft_cc)
sig_noise_mfcc = torch.fft.irfft(sig_noise_fft_mfcc)
# print("Removed Spec Shape: ", sig_DivSpecShape.shape)
# print("Removed Spec Shape audio: ", sig_noise.shape)

# plt.plot(sig_DivSpecShape[7])
# plt.plot(grain_db[9])
# plt.plot(dct.idct(cepstral_coeff)[9])
# plt.plot(np.abs(grain_fft)[9])
# plt.plot(inv_cepstral_coeff[9])
# plt.plot(inv_mfccs[9])
# plt.plot(sig_noise_cc[9])
# plt.plot(sig_noise_mfcc[9])
# for i in range(0, 425):
# plt.plot(sig_noise[424])
plt.savefig("test.png")
# TODO Plot my white noise and see what it looks like 
# DONE - Looks like there is a normalisation issue with the generated noise.

torch.manual_seed(10)

# Get the impulse response from the inverted cepstral coefficients
filter_ir = dsp.amp_to_impulse_response(inv_cepstral_coeff, l_grain)
filter_ir_mfcc = dsp.amp_to_impulse_response(inv_mfccs, l_grain)

# filter_ir = dsp.amp_to_impulse_response_w_phase(inv_cepstral_coeff, l_grain)

# Generate the noise grains
noise = utils.generate_noise_grains(bs, n_grains, l_grain, filter_ir.dtype, filter_ir.device, hop_ratio=0.25)


noise = noise.reshape(bs*n_grains, l_grain)
noise_fft = torch.fft.rfft(noise)

# Convolve the noise and impulse response 
audio = dsp.fft_convolve_no_pad(noise, filter_ir)
# audio = dsp.fft_convolve_no_pad(sig_noise_cc, filter_ir)
# Testing signal noise
# audio = dsp.fft_convolve_no_pad(sig_noise, filter_ir)
audio = audio.reshape(-1,n_grains,l_grain)


# Apply same window as previously to new filtered noise grains
ola_window = signal.hann(l_grain,sym=False)
ola_windows = torch.from_numpy(ola_window).unsqueeze(0).repeat(n_grains,1).type(torch.float32)
ola_windows[0,:l_grain//2] = ola_window[l_grain//2] # start of 1st grain is not windowed for preserving attacks
ola_windows[-1,l_grain//2:] = ola_window[l_grain//2] # end of last grain is not wondowed to preserving decays
ola_windows = ola_windows
audio = audio*(ola_windows.unsqueeze(0).repeat(bs,1,1))



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

# Apply simply normalisation, as is done in dataloader
# TODO Add this to training script also
audio_sum /= torch.max(torch.abs(audio_sum))
audio_sum *= 0.9
# audio = audio/torch.max(audio)

#calculate the spectral distacne
spec_dist = spectral_distances(sr=SAMPLE_RATE, device=DEVICE)
spec_loss = spec_dist(audio_sum, batch)

# print("Spectral Loss: ", spec_loss)

print(audio_sum[0].shape)
y_8k = librosa.resample(audio_sum[0].cpu().numpy(), orig_sr=SAMPLE_RATE, target_sr=16000)
print(y_8k.sum())


#for testing
for i in range(audio_sum.shape[0]):
    spec_loss = spec_dist(audio_sum[i], batch[i])
    print("Spectral Loss: ", spec_loss)
    torchaudio.save(f'/Users/adees/Code/neural_granular_synthesis/scripts/TDSP/test_audio/CC_testrecon_{i}.wav', audio_sum[i].unsqueeze(0).cpu(), SAMPLE_RATE)
    torchaudio.save(f"/Users/adees/Code/neural_granular_synthesis/scripts/TDSP/real_audio/CC_test_{i}.wav", batch[i].unsqueeze(0).cpu(), SAMPLE_RATE)
    fad_score = frechet.score(f"/Users/adees/Code/neural_granular_synthesis/scripts/TDSP/real_audio", f"/Users/adees/Code/neural_granular_synthesis/scripts/TDSP/test_audio", dtype="float32")
    print("FAD Score: ", fad_score)

for i in range(audio_sum.shape[0]):
    audio_sum_8k = torch.from_numpy(librosa.resample(audio_sum[i].cpu().numpy(), orig_sr=SAMPLE_RATE, target_sr=16000))
    batch_8k = torch.from_numpy(librosa.resample(batch[i].cpu().numpy(), orig_sr=SAMPLE_RATE, target_sr=16000))
    spec_loss = spec_dist(audio_sum_8k, batch_8k)
    print("Spectral Loss: ", spec_loss)
    # # batch_8k = batch_8k.to(torch.float16)
    # audio_sum_8k = audio_sum_8k.to(torch.float16)
    torchaudio.save(f'/Users/adees/Code/neural_granular_synthesis/scripts/TDSP/test_audio_16k/CC_testrecon_{i}.wav', (audio_sum_8k).unsqueeze(0).cpu(), 16000)
    torchaudio.save(f"/Users/adees/Code/neural_granular_synthesis/scripts/TDSP/real_audio_16k/CC_test_{i}.wav", (batch_8k).unsqueeze(0).cpu(), 16000)
    fad_score = frechet.score(f"/Users/adees/Code/neural_granular_synthesis/scripts/TDSP/real_audio_16k", f"/Users/adees/Code/neural_granular_synthesis/scripts/TDSP/test_audio_16k", dtype="float32")
    print("FAD Score: ", fad_score)
