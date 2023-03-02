import torch
import glob
import numpy as np
import pandas as pd
import torchaudio
import os

class FSDDSpectrogramDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, transform=None):
        super().__init__()
        
        self.root_dir = root_dir
        self.file_list = glob.glob(self.root_dir + "*")
        self.transform = transform

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):


        file_path = self.file_list[index]
        spectrogram = np.load(file_path)
        spectrogram = np.expand_dims(spectrogram, axis=0)
        spectrogram = torch.from_numpy(spectrogram)

        if self.transform:
            spectrogram = self.transform(spectrogram)

        return spectrogram, file_path
    
class UrbanSoundDataset(torch.utils.data.Dataset):

    def __init__(self, 
                    annotations_file, 
                    root_dir, 
                    transform, 
                    target_sample_rate,
                    num_samples,
                    device):

        self.device = device
        self.annotations = pd.read_csv(annotations_file)
        self.root_dir = root_dir
        self.transform = transform.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        # Cut / Pad so all signals are same duration
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)

        if self.transform:
            signal = self.transform(signal)

        return signal, label
    
    def _cut_if_necessary(self, signal):
        
        if signal.shape[1] > self.num_samples:
            signal = signal[: , :self.num_samples]
        
        return signal
    
    def _right_pad_if_necessary(self, signal):

        length_signal = signal.shape[1]

        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        
        return signal

    def _resample_if_necessary(self, signal, sr):

        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate).to(self.device)
            signal = resampler(signal)

        return signal

    def _mix_down_if_necessary(self, signal):

        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        
        return signal


    def _get_audio_sample_path(self, index):

        fold = f"fold{self.annotations.iloc[index, 5]}"
        path = os.path.join(self.root_dir, fold, 
                            self.annotations.iloc[index, 0])
                        
        return path
    
    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 7]