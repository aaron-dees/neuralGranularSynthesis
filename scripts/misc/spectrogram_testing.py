import sys
sys.path.append('../')

import torchaudio
from scripts.hyper_parameters_urbansound import *
from models.dataloaders import UrbanSoundDataset
import soundfile as sf

if __name__ == "__main__":

    
    signal, sr = torchaudio.load("/Users/adees/Code/neural_granular_synthesis/datasets/UrbanSound8K/audio/fold1/101415-3-0-2.wav")
    print(signal.shape)
    # torchaudio.save("/Users/adees/Code/neural_granular_synthesis/scripts/tester.wav", signal, sr)

    # create the mel spectrogram transform
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate = sr,
        n_fft = FRAME_SIZE,
        hop_length = HOP_LENGTH,
        n_mels = NUM_MELS
    )


    usd = UrbanSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, DATALOADER_DEVICE)
    counter = 0
    for mel_spec, albel in usd:
        counter+= 1
        spectrogram = torchaudio.transforms.InverseMelScale(sample_rate=SAMPLE_RATE, n_stft = (FRAME_SIZE//2+1), n_mels = NUM_MELS)(mel_spec)
        waveform = torchaudio.transforms.GriffinLim(n_fft=FRAME_SIZE, hop_length = HOP_LENGTH)(spectrogram)
        print(f"After transform {waveform.shape}")
        torchaudio.save(f"./audio_tests/usd_{albel}_{counter}.wav", waveform, SAMPLE_RATE)
        if counter>0:
            break


    # mel_spec = mel_spectrogram(signal)

    # spectrogram = torchaudio.transforms.InverseMelScale(sample_rate=SAMPLE_RATE, n_stft = (FRAME_SIZE//2+1), n_mels = NUM_MELS)(mel_spec)

    # waveform = torchaudio.transforms.GriffinLim(n_fft=FRAME_SIZE, hop_length = HOP_LENGTH)(spectrogram)

    # torchaudio.save("/Users/adees/Code/neural_granular_synthesis/scripts/tester.wav", waveform, sr)

