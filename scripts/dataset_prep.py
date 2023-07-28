import sys
sys.path.append('../')

from models.dataloaders.waveform_dataloaders import  ESC50WaveformDataset
import torch
import torchaudio
import pandas as pd
import os

ANNOTATIONS_FILE = "/Users/adees/Code/neural_granular_synthesis/datasets/ESC-50_SeaWaves/esc50_seaWaves.csv"
AUDIO_DIR = "/Users/adees/Code/neural_granular_synthesis/datasets/ESC-50_SeaWaves/audio"
SAVE_DIR = "/Users/adees/Code/neural_granular_synthesis/datasets/ESC-50_SeaWaves/audio/samples/2048/"
SAMPLE_RATE = 44100
NUM_SAMPLES = 44100*5
DATALOADER_DEVICE = torch.device('cpu')
SAMPLE_SIZE = 2048

if __name__ == "__main__":

    annotations = pd.read_csv(ANNOTATIONS_FILE)

    signals = []        
    for i in range(annotations.shape[0]):
        signal, sr = torchaudio.load(os.path.join(AUDIO_DIR, annotations.iloc[i, 0]))
        signals.append(signal)

    for i in range(len(signals)):
        for j in range(signals[i].shape[1]//SAMPLE_SIZE):
            sample = signals[i][:, j*SAMPLE_SIZE:j*SAMPLE_SIZE+SAMPLE_SIZE]
            save_path = f'{SAVE_DIR}{annotations.iloc[i, 0][:-4]}_{j}.wav'
            torchaudio.save(save_path, sample, sr)

    # for i in
    # path = os.path.join(self.root_dir, 
    #                         self.annotations.iloc[index, 0])

    # for i, data in enumerate(usd_waveforms):
    #     waveform, label = data 
    #     torchaudio.save(f"./audio_tests/original/orig_{label[:-4]}.wav", waveform, SAMPLE_RATE)

    # print("Loaded waveforms")