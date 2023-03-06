import torch
import math

def mod_sigmoid(x):
    return 2 * torch.sigmoid(x)**(math.log(10)) + 1e-7

def safe_log(x, eps=1e-7):
    return torch.log(x + eps)

def noise_filtering(filter_coeffs,filter_window):
    N = filter_coeffs.shape[0]
    num_samples = (filter_coeffs.shape[1]-1)*2
    dtype = filter_coeffs.dtype
    # create impulse response
    # torch.complex is not implemented on MPS, use CPU
    filter_coeffs = torch.complex(filter_coeffs,torch.zeros_like(filter_coeffs))
    # Inverting filter coefficients from fourier domain --> tmie domain for windowing
    filter_ir = torch.fft.irfft(filter_coeffs)
    # Apply windowing
    filter_ir = filter_ir*filter_window.unsqueeze(0).repeat(N,1)
    # Apply fft shift (how does this work in time domain?)
    filter_ir = torch.fft.fftshift(filter_ir,dim=-1)

    # convolve with noise signal
    # Create noise, why doe we multiply by 2 and subtract 1 here
    noise = torch.rand(N, num_samples, dtype=dtype, device=filter_coeffs.device)*2-1
    # Transform noise and impulse response filters into fourier domain
    S_noise = torch.fft.rfft(noise,dim=1)
    S_filter = torch.fft.rfft(filter_ir,dim=1)
    # Conv (multiply in fourier domain)
    S = torch.mul(S_noise,S_filter)
    # Invert back into time domain to get audio
    audio = torch.fft.irfft(S)

    return audio