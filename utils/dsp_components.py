import torch
import math
import matplotlib.pyplot as plt

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
def noise_filtering(filter_coeffs,filter_window):
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
    # convolve with noise signal
    # Create noise, why doe we multiply by 2 and subtract 1 here
    # noise = torch.rand(N, num_samples, dtype=dtype, device=filter_coeffs.device)*2-1
    single_noise = torch.rand(num_samples, dtype=dtype, device=filter_coeffs.device)*2-1
    noise = single_noise.unsqueeze(0).repeat(N, 1)
    # Transform noise and impulse response filters into fourier domain
    S_noise = torch.fft.rfft(noise,dim=1)
    S_filter = torch.fft.rfft(filter_ir,dim=1)
    # Conv (multiply in fourier domain)
    S = torch.mul(S_noise,S_filter)
    # Invert back into time domain to get audio
    audio = torch.fft.irfft(S)

    # Note that overlapp and add is used here in DDSP, 
    # but this is because they are usung a bunch of audio frames
    # do they doe something similar in Neural Gran Synth

    return audio