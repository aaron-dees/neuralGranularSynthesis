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
import librosa.display
import torch_dct as dct
from models.loss_functions import spectral_distances
from frechet_audio_distance import FrechetAudioDistance

import utils.dsp_components as dsp
import utils.utilities as utils

torch.manual_seed(10)


# to use `vggish`
frechet = FrechetAudioDistance(
    model_name="vggish",
    sample_rate=16000,
    use_pca=False, 
    use_activation=False,
    verbose=False
)

# Torch dataloader bits
train_dataloader,val_dataloader,dataset,tar_l,n_grains,l_grain,hop_size,classes = make_audio_dataloaders(data_dir=AUDIO_DIR,classes=["sea_waves"],sr=SAMPLE_RATE,silent_reject=[0,0],amplitude_norm=True,batch_size=BATCH_SIZE,hop_ratio=HOP_SIZE_RATIO, tar_l=TARGET_LENGTH,l_grain=GRAIN_LENGTH,high_pass_freq=HIGH_PASS_FREQ,num_workers=0, center_pad=CENTER_PADDING)
test_set = torch.utils.data.Subset(dataset, range(0,TEST_SIZE))
test_dataloader = torch.utils.data.DataLoader(test_set, batch_size = TEST_SIZE, shuffle=False, num_workers=0)

# Get grains and add windowing
for batch, labels in test_dataloader:

    # use librosa to get log mag spec
    # print(batch.shape)
    if(CENTER_PADDING):
        # stft_audio = librosa.stft(batch[:, 256:-256].squeeze().cpu().numpy(), n_fft = l_grain, hop_length = hop_size, center=True)
        ola_window = torch.from_numpy(signal.hann(l_grain,sym=False)).type(torch.float32)
        stft_audio = torch.stft(batch[:, 256:-256].squeeze().cpu(), n_fft = l_grain, hop_length = hop_size, window=ola_window, center=True, return_complex=True, pad_mode="constant")
    else:
        stft_audio = librosa.stft(batch.squeeze().cpu().numpy(), n_fft = l_grain, hop_length = hop_size, center=True)
    print("STFT: ", stft_audio.shape)

    # y_inv = librosa.griffinlim(np.abs(stft_audio))
    transform = torchaudio.transforms.GriffinLim(n_fft=l_grain, hop_length=hop_size, power=1)
    y_inv = transform(torch.abs(stft_audio))
    # y_inv = transform(torch.abs(torch.from_numpy(stft_audio)))
    torchaudio.save(f'/Users/adees/Code/neural_granular_synthesis/scripts/TDSP/griffinLimAudio/reconstructed_stftGriffLim.wav',y_inv.unsqueeze(0).cpu(), SAMPLE_RATE)
   
   
    # y_log_audio = librosa.power_to_db(stft_audio)
    y_log_audio = 20*dsp.safe_log10(torch.abs(stft_audio))
    # y_log_audio = 20*dsp.safe_log10(torch.abs(torch.from_numpy(stft_audio)))
    
    # cepstral_coeff = dct.dct(torch.from_numpy(y_log_audio).permute(1,0))
    cepstral_coeff = dct.dct(y_log_audio.permute(1,0))
    cepstral_coeff[:,NUM_CC:] = 0
    inv_cepstral_coeff_librosa = 10**(dct.idct(cepstral_coeff) / 20)
    # plt.figure()
    # librosa.display.specshow(inv_cepstral_coeff_librosa.permute(1,0).cpu().numpy(), n_fft=l_grain, hop_length=hop_size, sr=SAMPLE_RATE, x_axis='time', y_axis='log')
    # plt.colorbar()
    # plt.savefig("test.png")
    # plt.plot(inv_cepstral_coeff_librosa[9])


    # test librosa lpc
    # print("Batch shape: ", batch.shape)
    # lpc = librosa.lpc(batch.cpu().numpy(), order=128)
    # print(lpc.shape)
    # plt.plot(batch.cpu().numpy()[0])
    # plt.plot(lpc[0])
    # plt.savefig("test.png")

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

# ----- CC Inversion -----
grain_fft = torch.fft.rfft(mb_grains)
# grain_db = 20*dsp.safe_log10(np.abs(grain_fft))
#GriffinLim test
transform = torchaudio.transforms.GriffinLim(n_fft=l_grain, hop_length=hop_size, power=1)
grain_inv = transform(torch.abs(grain_fft).permute(1,0))
print(grain_inv.shape)
# torchaudio.save(f'/Users/adees/Code/neural_granular_synthesis/scripts/TDSP/griffinLimAudio/reconstructed_grainGriffLim.wav',grain_inv.unsqueeze(0).cpu(), SAMPLE_RATE)

grain_db = 20*dsp.safe_log10(np.abs(grain_fft))
cepstral_coeff = dct.dct(grain_db)
# Cut of some of the shape
cepstral_coeff[:,NUM_CC:] = 0
# Invert the cepstral coefficitents and scale from db -> power scale.
inv_cepstral_coeff = 10**(dct.idct(cepstral_coeff) / 20)

# ----- MFCC Invesion -----
grain_fft_perm = grain_fft.permute(1,0)
print()
grain_mel= dsp.safe_log10(torch.from_numpy((librosa.feature.melspectrogram(S=np.abs(grain_fft_perm)**2, sr=SAMPLE_RATE, n_fft=GRAIN_LENGTH, n_mels=NUM_MELS))))
# grain_mel = grain_mel.permute(1,0)
mfccs = dct.dct(grain_mel)
mfccs[:,200:] = 0
inv_mfccs = dct.idct(mfccs).cpu().numpy()
# inv_mfccs = inv_mfccs.permute(1,0).cpu().numpy()
inv_mfccs = librosa.feature.inverse.mel_to_stft(M=10**inv_mfccs, sr=SAMPLE_RATE, n_fft=GRAIN_LENGTH)
inv_mfccs = torch.from_numpy(inv_mfccs).permute(1,0)
# plt.figure()
# librosa.display.specshow(inv_mfccs.permute(1,0).cpu().numpy(), n_fft=l_grain, hop_length=hop_size, sr=SAMPLE_RATE, x_axis='time', y_axis='log')
# plt.colorbar()
# plt.savefig("test.png")

# inv_mfccs[inv_mfccs == 0] = 1e-10

# print("test_mfcc: ", inv_mfccs.sum())
# print("test_cc: ", inv_cepstral_coeff.sum())

# print(inv_mfccs[0, -50:])
# print(inv_cepstral_coeff[0, -50:])


# Divide the original signal by the inverted CC's
sig_noise_fft_cc = grain_fft / inv_cepstral_coeff
# have to add small amount here to avoid nan's
sig_noise_fft_mfcc = grain_fft / (inv_mfccs)

# print("Tester: ", inv_mfccs)

# Original Signal Noise
sig_noise_cc = torch.fft.irfft(sig_noise_fft_cc)
sig_noise_mfcc = torch.fft.irfft(sig_noise_fft_mfcc)

# plt.plot(sig_noise_fft_cc[9])
# plt.plot(np.abs(grain_fft[9]))
# plt.plot(inv_mfccs[9])
# plt.plot(inv_cepstral_coeff[9])
# plt.figure()
# # librosa.display.specshow(dsp.safe_log10(torch.abs(sig_noise_fft_cc)**2).cpu().numpy())
# librosa.display.specshow(dsp.safe_log10(torch.abs(noise)**2).cpu().numpy())
# plt.savefig("test.png")

# Get the impulse response from the inverted cepstral coefficients
filter_ir = dsp.amp_to_impulse_response(inv_cepstral_coeff, l_grain)
filter_ir_mfcc = dsp.amp_to_impulse_response(inv_mfccs+1e-7, l_grain)

# Generate the noise grains
noise = utils.generate_noise_grains(bs, n_grains, l_grain, torch.float32, DEVICE, hop_ratio=0.25)
noise = noise.reshape(bs*n_grains, l_grain)
# noise_fft = torch.fft.rfft(noise)
# print("Noise: ", noise.shape)
# torchaudio.save(f'/Users/adees/Code/neural_granular_synthesis/scripts/TDSP/noise.wav', noise.unsqueeze(0).cpu(), SAMPLE_RATE)
# torchaudio.save(f"/Users/adees/Code/neural_granular_synthesis/scripts/TDSP/sig_noise_cc", sig_noise_cc.unsqueeze(0).cpu(), SAMPLE_RATE)

# plt.figure()
# librosa.display.specshow(dsp.safe_log10(torch.abs(sig_noise_fft_cc*10)**2).cpu().numpy())
# # librosa.display.specshow(dsp.safe_log10(torch.abs(noise)**2).cpu().numpy())
# # plt.plot(sig_noise_cc[7])
# plt.plot(torch.fft.rfft(noise[9]))
# plt.savefig("test.png")

# Convolve the noise and impulse response 
# audio = dsp.fft_convolve_no_pad(noise, filter_ir)
audio = dsp.fft_convolve_no_pad_2(noise, inv_mfccs)
# audio = dsp.fft_convolve_no_pad_2(noise, inv_cepstral_coeff)

# padding test - TODO Ask Sean about this.
# audio = dsp.fft_convolve(sig_noise_cc, filter_ir)
# audio = dsp.fft_convolve_no_pad(sig_noise_cc, filter_ir)

# audio = dsp.fft_convolve_no_pad(sig_noise_cc, filter_ir)
# Testing signal noise
audio = audio.reshape(-1,n_grains,l_grain)


# Apply same window as previously to new filtered noise grains
ola_window = signal.hann(l_grain,sym=False)
ola_windows = torch.from_numpy(ola_window).unsqueeze(0).repeat(n_grains,1).type(torch.float32)
ola_windows[0,:l_grain//2] = ola_window[l_grain//2] # start of 1st grain is not windowed for preserving attacks
ola_windows[-1,l_grain//2:] = ola_window[l_grain//2] # end of last grain is not wondowed to preserving decays
ola_windows = ola_windows
audio = audio*(ola_windows.unsqueeze(0).repeat(bs,1,1))

noise = noise*(ola_windows.unsqueeze(0).repeat(bs,1,1))

# Overlap add folder, folds and reshapes audio into target dimensions, so [bs, tar_l]
# This is essentially folding and adding the overlapping grains back into original sample size, based on the given hop size.
# Note that shape is changed here, so that input tensor is of shape [bs, channel X (kernel_size), L],
# since kernel size is l_grain, this is needed in the second dimension.
ola_folder = torch.nn.Fold((tar_l,1),(l_grain,1),stride=(hop_size,1))
audio_sum = ola_folder(audio.permute(0,2,1)).squeeze()

# print("Shape: ", audio.shape)
# print("Shape: ", noise.shape)
# print("Shape: ", sig_noise_cc.shape)
noise_sum = ola_folder(noise.permute(0,2,1)).squeeze()
sig_noise_sum = ola_folder(sig_noise_cc.unsqueeze(0).permute(0,2,1)).squeeze()
print("TEST: ", sig_noise_sum.sum())
sig_noise_sum_mel = ola_folder(sig_noise_mfcc.unsqueeze(0).permute(0,2,1)).squeeze()
print("TEST: ", sig_noise_sum_mel.sum())


# Normalise the energy values across the audio samples
if NORMALIZE_OLA:
    # Normalises based on number of overlapping grains used in folding per point in time.
    unfolder = torch.nn.Unfold((l_grain,1),stride=(hop_size,1))
    input_ones = torch.ones(1,1,tar_l,1)
    ola_divisor = ola_folder(unfolder(input_ones)).squeeze()
    ola_divisor = ola_divisor
    audio_sum = audio_sum/ola_divisor.unsqueeze(0).repeat(bs,1)
    noise_sum = noise_sum/ola_divisor.unsqueeze(0).repeat(bs,1)
    sig_noise_sum = sig_noise_sum/ola_divisor.unsqueeze(0).repeat(bs,1)
    sig_noise_sum_mel = sig_noise_sum_mel/ola_divisor.unsqueeze(0).repeat(bs,1)
else:
    audio_sum = audio_sum.unsqueeze(0)

# Apply simply normalisation, as is done in dataloader
# TODO Add this to training script also
# audio_sum /= torch.max(torch.abs(audio_sum))
# audio_sum *= 0.9
noise_sum /= torch.max(torch.abs(noise_sum))
noise_sum *= 0.9
# sig_noise_sum /= torch.max(torch.abs(sig_noise_sum))
# sig_noise_sum *= 0.9

# Normlize based on energy
audio_sum = audio_sum * (torch.sqrt((batch**2).sum()) / torch.sqrt((audio_sum**2).sum()))

sig_noise_sum = sig_noise_sum * (torch.sqrt((noise_sum**2).sum()) / torch.sqrt((sig_noise_sum**2).sum()))

sig_noise_sum_mel = sig_noise_sum_mel * (torch.sqrt((noise_sum**2).sum()) / torch.sqrt((sig_noise_sum_mel**2).sum()))
# audio = audio/torch.max(audio)
print("Shape: ", noise_sum.shape)

torchaudio.save(f'/Users/adees/Code/neural_granular_synthesis/scripts/TDSP/noise.wav', noise_sum.cpu(), SAMPLE_RATE)
torchaudio.save(f"/Users/adees/Code/neural_granular_synthesis/scripts/TDSP/sig_noise_cc.wav", sig_noise_sum.cpu(), SAMPLE_RATE)
torchaudio.save(f"/Users/adees/Code/neural_granular_synthesis/scripts/TDSP/sig_noise_mel.wav", sig_noise_sum_mel.cpu(), SAMPLE_RATE)

#calculate the spectral distacne
spec_dist = spectral_distances(sr=SAMPLE_RATE, device=DEVICE)

# clip the audio to remove the padded areas
if(CENTER_PADDING):
    batch=batch[:,256:-256]
    audio_sum=audio_sum[:,256:-256]

#for testing
for i in range(audio_sum.shape[0]):
    spec_loss = spec_dist(audio_sum[i], batch[i])
    print("Spectral Loss: ", spec_loss)
    torchaudio.save(f'/Users/adees/Code/neural_granular_synthesis/scripts/TDSP/test_audio/reconstructed_{i}.wav', audio_sum[i].unsqueeze(0).cpu(), SAMPLE_RATE)
    torchaudio.save(f"/Users/adees/Code/neural_granular_synthesis/scripts/TDSP/real_audio/original_{i}.wav", batch[i].unsqueeze(0).cpu(), SAMPLE_RATE)
    fad_score = frechet.score(f"/Users/adees/Code/neural_granular_synthesis/scripts/TDSP/real_audio", f"/Users/adees/Code/neural_granular_synthesis/scripts/TDSP/test_audio", dtype="float32")
    print("FAD Score: ", fad_score)

# Additional loss based on griffinLim
fad_score = frechet.score(f"/Users/adees/Code/neural_granular_synthesis/scripts/TDSP/real_audio", f"/Users/adees/Code/neural_granular_synthesis/scripts/TDSP/griffinLimAudio", dtype="float32")
spec_loss = spec_dist(y_inv, batch[0])
print("GriffinLim Spectral Loss: ", spec_loss)
print("GriffinLim FAD Score: ", fad_score)

