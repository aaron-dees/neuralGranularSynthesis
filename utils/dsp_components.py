import torch
import math
import matplotlib.pyplot as plt
# from utils.utilities import generate_noise_grains, generate_noise_grains_stft
import utils.utilities as utils
import numpy as np
from scipy import fftpack


##################
#   Modified Sigmoid
#
#   https://arxiv.org/pdf/2001.04643.pdf - Sec B.5
#   We force the amplitudes, harmonic distributions, and filtered noise magnitudes to be non-negative
#   by applying a sigmoid nonlinearity to network outputs. We find a slight improvement in traning
#    stability by modifying the sigmoid to have a scaled output, larger slope by exponentiating, and
#   threshold at a minimum value, as seen in below retunr statement.
def mod_sigmoid(x):
    # return 4 * torch.sigmoid(x)**(math.log(10)) + 1e-7
    return 2 * torch.sigmoid(x)**(math.log(10)) + 1e-7
    # return 2 * torch.sigmoid(x)**(math.log(10)) + 1e-18

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
    # audio = fft_convolve_no_pad(noise, filter_ir)
    # Note that we are not using Impulse Response here
    audio = fft_convolve_no_pad_2(noise, filter_coeffs)

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

def frame(signal, frame_length, frame_step, pad_end=False, pad_value=0, axis=-1):
    """
    equivalent of tf.signal.frame
    """
    pad_size = 0
    signal_length = signal.shape[axis]
    if pad_end:
        frames_overlap = frame_length - frame_step
        rest_samples = np.abs(signal_length - frames_overlap) % np.abs(frame_length - frames_overlap)
        pad_size = int(frame_length - rest_samples)
        if pad_size != 0:
            pad_axis = [0] * signal.ndim
            pad_axis[axis] = pad_size
            signal = torch.nn.functional.pad(signal, pad_axis, "constant", pad_value)
    frames=signal.unfold(axis, frame_length, frame_step)
    return frames

def get_fft_size(frame_size: int, ir_size: int, power_of_2: bool = True) -> int:
  """Calculate final size for efficient FFT.

  Args:
    frame_size: Size of the audio frame.
    ir_size: Size of the convolving impulse response.
    power_of_2: Constrain to be a power of 2. If False, allow other 5-smooth
      numbers. TPU requires power of 2, while GPU is more flexible.

  Returns:
    fft_size: Size for efficient FFT.
  """
  convolved_frame_size = ir_size + frame_size - 1
  if power_of_2:
    # Next power of 2.
    fft_size = int(2**np.ceil(np.log2(convolved_frame_size)))
  else:
    fft_size = int(fftpack.helper.next_fast_len(convolved_frame_size))
  return fft_size

def crop_and_compensate_delay(audio, audio_size, ir_size,
                              padding,
                              delay_compensation):
  """Crop audio output from convolution to compensate for group delay.

  Args:
    audio: Audio after convolution. Tensor of shape [batch, time_steps].
    audio_size: Initial size of the audio before convolution.
    ir_size: Size of the convolving impulse response.
    padding: Either 'valid' or 'same'. For 'same' the final output to be the
      same size as the input audio (audio_timesteps). For 'valid' the audio is
      extended to include the tail of the impulse response (audio_timesteps +
      ir_timesteps - 1).
    delay_compensation: Samples to crop from start of output audio to compensate
      for group delay of the impulse response. If delay_compensation < 0 it
      defaults to automatically calculating a constant group delay of the
      windowed linear phase filter from frequency_impulse_response().

  Returns:
    Tensor of cropped and shifted audio.

  Raises:
    ValueError: If padding is not either 'valid' or 'same'.
  """
  # Crop the output.
  if padding == 'valid':
    crop_size = ir_size + audio_size - 1
  elif padding == 'same':
    crop_size = audio_size
  else:
    raise ValueError('Padding must be \'valid\' or \'same\', instead '
                     'of {}.'.format(padding))

  # Compensate for the group delay of the filter by trimming the front.
  # For an impulse response produced by frequency_impulse_response(),
  # the group delay is constant because the filter is linear phase.
  total_size = int(audio.shape[-1])
  crop = total_size - crop_size
  start = ((ir_size - 1) // 2 -
           1 if delay_compensation < 0 else delay_compensation)
  end = crop - start
  return audio[:, start:-end]

def fft_convolve_ddsp(signal, impulse_response):

    if(len(signal.shape)!=2):
        raise ValueError("Signal must be only 2 dimensions")
    # Cut audio into frames.
    audio_size = signal.shape[-1]
    n_ir_frames = impulse_response.shape[1]
    ir_size = impulse_response.shape[-1]
    frame_size = int(np.ceil(audio_size / n_ir_frames))
    hop_size = frame_size
    audio_frames = frame(signal, frame_size, hop_size, pad_end=True)
    ola_folder = torch.nn.Fold((audio_size+hop_size, 1),(hop_size,1), stride=(hop_size,1))
    test = ola_folder(audio_frames.permute(0,2,1)).squeeze(0).squeeze(-1)

    # Check number of frames match
    n_audio_frames = int(audio_frames.shape[1])
    # print(impulse_response.shape)
    if n_audio_frames != n_ir_frames:
        raise ValueError(
            'Number of Audio frames ({}) and impulse response frames ({}) do not '
            'match. For small hop size = ceil(audio_size / n_ir_frames), '
            'number of impulse response frames must be a multiple of the audio '
            'size.'.format(n_audio_frames, n_ir_frames))
    
    # Pad and FFT the audio and impulse responses.
    fft_size = get_fft_size(frame_size, ir_size, power_of_2=True)
    audio_fft = torch.fft.rfft(audio_frames, fft_size)
    ir_fft = torch.fft.rfft(impulse_response, fft_size)

    # signal = torch.nn.functional.pad(signal, (0, signal.shape[-1]))
    # kernel = torch.nn.functional.pad(kernel, (kernel.shape[-1], 0))

    # NOTE Should I really be using ifft here since we want to keep the phase of the noise. 
    audio_frames_out = torch.fft.irfft(audio_fft * ir_fft)
    #HACK - Pad output size, I've test this and it seems to be the way to do it, then we can clip end off.
    ola_folder = torch.nn.Fold((audio_size+fft_size, 1),(fft_size,1), stride=(hop_size,1))
    output = ola_folder(audio_frames_out.permute(0,2,1)).squeeze(0).squeeze(-1)
    if(len(output.shape)>2):
       output = output.squeeze(1)

    delay_compensation = -1
    padding = "same"
    cropped_out = crop_and_compensate_delay(output, audio_size, ir_size, padding,
                                   delay_compensation)


    return cropped_out

def fft_convolve(signal, kernel):

    signal = torch.nn.functional.pad(signal, (0, signal.shape[-1]))
    kernel = torch.nn.functional.pad(kernel, (kernel.shape[-1], 0))

    # NOTE Should I really be using ifft here since we want to keep the phase of the noise. 
    output = torch.fft.irfft(torch.fft.rfft(signal) * torch.fft.rfft(kernel))
    # plt.plot((torch.fft.rfft(signal) * torch.fft.rfft(kernel))[7])
    # plt.savefig("test_2.png")
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

    # plt.plot((torch.fft.rfft(signal) * torch.fft.rfft(kernel))[7])
    # plt.savefig("test_1.png")


    return output

def fft_convolve_no_pad_2(signal, kernel):


    output = torch.fft.irfft(torch.fft.rfft(signal) * kernel)
    # plt.plot(torch.abs((torch.fft.rfft(signal) * kernel)[9]))
    # plt.plot(np.abs(grain_fft[9]))
    # plt.plot(inv_mfccs[7])
    # plt.plot(inv_cepstral_coeff[9])
    # plt.figure()
    # # librosa.display.specshow(dsp.safe_log10(torch.abs(sig_noise_fft_cc)**2).cpu().numpy())
    # librosa.display.specshow(dsp.safe_log10(torch.abs(noise)**2).cpu().numpy())
    # plt.savefig("test.png")


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
