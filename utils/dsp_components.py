import torch
import math
import matplotlib.pyplot as plt
from utils.utilities import generate_noise_grains, frame, get_fft_size
import numpy as np

##################
#   Modified Sigmoid
#
#   https://arxiv.org/pdf/2001.04643.pdf - Sec B.5
#   We force the amplitudes, harmonic distributions, and filtered noise magnitudes to be non-negative
#   by applying a sigmoid nonlinearity to network outputs. We find a slight improvement in traning
#    stability by modifying the sigmoid to have a scaled output, larger slope by exponentiating, and
#   threshold at a minimum value, as seen in below retunr statement.
def mod_sigmoid(x):
    return 2 * torch.sigmoid(x)**(math.log(10)) + 1e-7

def safe_log(x, eps=1e-7):
    return torch.log(x + eps)

###################
#   Noise Filtering - Follows same as DDSP repo
#   https://github.com/magenta/ddsp/blob/main/ddsp/synths.py#L181
#       Takes fourier domain filter coefficients and a filter window and returns filtered uniform noise
#           - Calculate parameters, N, num_samples
#           - Invert filter coefficients from fourier domain -> time domain
#           - Applywindowing by multiplying time domain filter coefficients by filter (FFT, Hann etc) window
#           - Apply FFT shift on filter coefficient in time domain
#           - Create noise signal from uniform noise
#           - Transform noise signal and windowed filter coefficients into  fourier domain
#           - Convolve the signal, by multiplying them in the fourier domain
#           - Perform inverse fourier transform of new signal to get audio signal
def noise_filtering(filter_coeffs,filter_window, n_grains, l_grain):


    # freq_impulse_response in ddsp
    N = filter_coeffs.shape[0]
    # get number of sample based on number of freq bins
    num_samples = (filter_coeffs.shape[1]-1)*2
    dtype = filter_coeffs.dtype
    # create impulse response
    # torch.complex is not implemented on MPS, use CPU
    filter_coeffs = torch.complex(filter_coeffs,torch.zeros_like(filter_coeffs))
    # Inverting filter coefficients from fourier domain --> tmie domain for windowing
    filter_ir = torch.fft.irfft(filter_coeffs)
    # Apply windowing
    filter_ir = filter_ir*filter_window.unsqueeze(0).repeat(N,1)
    # ir = filter_ir[0].detach().cpu().numpy()
    # plt.plot(ir)
    # plt.savefig("windowed_impulse_response.png")

    # Apply fft shift 
    # Question - Why are we doing this and what is it doing, can see that it is done in DDSP
    
    filter_ir = torch.fft.fftshift(filter_ir,dim=-1)

    # Create noise, why doe we multiply by 2 and subtract 1 here
    bs = filter_ir.reshape(-1,n_grains,l_grain).shape[0]
    noise = generate_noise_grains(bs, n_grains, l_grain, dtype, filter_coeffs.device, hop_ratio=0.25)
    noise = noise.reshape(bs*n_grains, l_grain)

    # convolve with noise signal - fft_convolve in ddsp
    # print("Filter shape: ", filter_ir.shape)
    filter_ir = filter_ir.unsqueeze(1)
    # print("Filter shape: ", filter_ir.shape)
    batch_size_ir = filter_ir.shape[0]
    n_ir_frames = filter_ir.shape[1]
    ir_size = filter_ir.shape[2]
    # print(n_ir_frames)

    frame_size = int(np.ceil(l_grain / n_ir_frames))
    hop_size = frame_size
    # print(hop_size)
    noise_frames = frame(noise, frame_size, hop_size, pad_end=True)

    n_audio_frames = int(noise_frames.shape[1])
    if n_audio_frames != n_ir_frames:
        print("ERROR")
    
    fft_size = get_fft_size(frame_size, ir_size, power_of_2=True)

    # print("FFT Size: ", fft_size)

    # audio_fft = tf.signal.rfft(audio_frames, [fft_size])
    # ir_fft = tf.signal.rfft(impulse_response, [fft_size])

    # Old noise functions
    # noise = torch.rand(N, num_samples, dtype=dtype, device=filter_coeffs.device)*2-1

    # Transform noise and impulse response filters into fourier domain
    # print(noise_frames.shape)
    # print(filter_ir.shape)
    S_noise = torch.fft.rfft(noise_frames,fft_size).squeeze()
    S_filter = torch.fft.rfft(filter_ir,fft_size).squeeze()
    # print("Noise: ", S_noise.shape)
    # print("Filter: ", S_filter.shape)
    # print("Noise: ", S_noise.dtype)
    # print("Filter: ", S_filter.dtype)
    # Conv (multiply in fourier domain)
    S = torch.mul(S_noise,S_filter)
    # Invert back into time domain to get audio
    audio = torch.fft.irfft(S)
    # print("Audio shape: ", audio.shape)

    # Note that overlapp and add is used here in DDSP, 
    # but this is because they are usung a bunch of audio frames
    # do they doe something similar in Neural Gran Synth

    return audio
