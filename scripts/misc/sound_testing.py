import sys
sys.path.append('../')

from models.dataloaders.waveform_dataloaders import  ESC50WaveformDataset
import torch
import torchaudio

ANNOTATIONS_FILE = "/Users/adees/Code/neural_granular_synthesis/datasets/ESC-50_FireCrackling/esc50_fire.csv"
AUDIO_DIR = "/Users/adees/Code/neural_granular_synthesis/datasets/ESC-50_FireCrackling/audio"
SAMPLE_RATE = 44100
NUM_SAMPLES = 2048*43
DATALOADER_DEVICE = torch.device('cpu')

if __name__ == "__main__":

    usd_waveforms = ESC50WaveformDataset(ANNOTATIONS_FILE, AUDIO_DIR, None, SAMPLE_RATE, NUM_SAMPLES, DATALOADER_DEVICE)

    for i, data in enumerate(usd_waveforms):
        waveform, label = data 
        torchaudio.save(f"./audio_tests/original/orig_{label[:-4]}.wav", waveform, SAMPLE_RATE)

    print("Loaded waveforms")