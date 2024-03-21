import torch
import math
import matplotlib.pyplot as plt
# from utils.utilities import generate_noise_grains, generate_noise_grains_stft
import utils.utilities as utils

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

def safe_log10(x, eps=1e-7):
    return torch.log10(x + eps)

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
def noise_filtering(filter_coeffs,filter_window, n_grains, l_grain, hop_ratio):
    # N = filter_coeffs.shape[0]
    # get number of sample based on number of freq bins
    # num_samples = (filter_coeffs.shape[1]-1)*2
    dtype = filter_coeffs.dtype
    # # create impulse response
    # # torch.complex is not implemented on MPS, use CPU
    # filter_coeffs = torch.complex(filter_coeffs,torch.zeros_like(filter_coeffs))
    # # Inverting filter coefficients from fourier domain --> tmie domain for windowing
    # filter_ir = torch.fft.irfft(filter_coeffs)
    # # Apply windowing
    # filter_ir = filter_ir*filter_window.unsqueeze(0).repeat(N,1)
    # # ir = filter_ir[0].detach().cpu().numpy()
    # # plt.plot(ir)
    # # plt.savefig("windowed_impulse_response.png")

    # # Apply fft shift 
    # # Question - Why are we doing this and what is it doing, can see that it is done in DDSP
    
    # filter_ir = torch.fft.fftshift(filter_ir,dim=-1)
    # # convolve with noise signal
    # # Create noise, why doe we multiply by 2 and subtract 1 here

    filter_ir = amp_to_impulse_response(filter_coeffs, l_grain)

    bs = filter_ir.reshape(-1,n_grains,l_grain).shape[0]
    
    noise = utils.generate_noise_grains(bs, n_grains, l_grain, dtype, filter_coeffs.device, hop_ratio=hop_ratio)
    noise = noise.reshape(bs*n_grains, l_grain)

    
    # Old noise functions
    # noise = torch.rand(N, num_samples, dtype=dtype, device=filter_coeffs.device)*2-1

    # audio = fft_convolve(noise, filter_ir)
    audio = fft_convolve_no_pad(noise, filter_ir)

    # Transform noise and impulse response filters into fourier domain
    # S_noise = torch.fft.rfft(noise,dim=1)
    # S_filter = torch.fft.rfft(filter_ir,dim=1)
    # # Conv (multiply in fourier domain)
    # S = torch.mul(S_noise,S_filter)
    # # Invert back into time domain to get audio
    # audio = torch.fft.irfft(S)

    # Note that overlapp and add is used here in DDSP, 
    # but this is because they are usung a bunch of audio frames
    # do they doe something similar in Neural Gran Synth

    return audio

def fft_convolve(signal, kernel):

    signal = torch.nn.functional.pad(signal, (0, signal.shape[-1]))
    kernel = torch.nn.functional.pad(kernel, (kernel.shape[-1], 0))

    # NOTE Should I really be using ifft here since we want to keep the phase of the noise. 
    output = torch.fft.irfft(torch.fft.rfft(signal) * torch.fft.rfft(kernel))
    output = output[..., output.shape[-1] // 2:]


    return output

def fft_convolve_2(signal, kernel):

    signal = torch.nn.functional.pad(signal, (0, signal.shape[-1]))
    kernel = torch.nn.functional.pad(kernel, (kernel.shape[-1], 0))

    output = torch.fft.rfft(signal) * torch.fft.rfft(kernel)
    output = output[..., output.shape[-1] // 2:]


    return output

def fft_convolve_no_pad(signal, kernel):

    output = torch.fft.irfft(torch.fft.rfft(signal) * torch.fft.rfft(kernel))


    return output

def amp_to_impulse_response(amp, target_size):

    amp = torch.stack([amp, torch.zeros_like(amp)], -1)
    amp = torch.view_as_complex(amp)
    amp = torch.fft.irfft(amp)

    filter_size = amp.shape[-1]

    amp = torch.roll(amp, filter_size // 2, -1)
    win = torch.hann_window(filter_size, dtype=amp.dtype, device=amp.device)

    amp = amp * win

    amp = torch.nn.functional.pad(amp, (0, int(target_size) - int(filter_size)))
    amp = torch.roll(amp, -filter_size // 2, -1)

    return amp

# When padding amp with complex component already in it.
def amp_to_impulse_response_w_phase(amp, target_size):

    amp = torch.fft.irfft(amp)

    filter_size = amp.shape[-1]

    amp = torch.roll(amp, filter_size // 2, -1)
    win = torch.hann_window(filter_size, dtype=amp.dtype, device=amp.device)

    amp = amp * win

    amp = torch.nn.functional.pad(amp, (0, int(target_size) - int(filter_size)))
    amp = torch.roll(amp, -filter_size // 2, -1)

    return amp
