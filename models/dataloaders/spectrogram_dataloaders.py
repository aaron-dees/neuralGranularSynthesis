
import torch
import glob
import numpy as np
import pandas as pd
import torchaudio
import os
import soundfile as sf
import librosa


class WaveformDataset(torch.utils.data.Dataset):

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
        self.transform = None
        if transform:
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
        return self.annotations.iloc[index, 6]
    
class ESC50WaveformDataset(torch.utils.data.Dataset):

    def __init__(self, 
                    annotations_file, 
                    root_dir, 
                    transform, 
                    target_sample_rate,
                    num_samples,
                    device):

        self.device = device
        # self.annotations = pd.read_csv(annotations_file)
        self.root_dir = root_dir
        self.transform = None
        if transform:
            self.transform = transform.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.filenames = os.listdir(root_dir)
        # print(self.filenames)
        

    def __len__(self):
        # return len(self.annotations)
        return len(self.filenames)
    
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

        # fold = f"fold{self.annotations.iloc[index, 5]}"
        # path = os.path.join(self.root_dir, 
        #                     self.annotations.iloc[index, 0])
        path = os.path.join(self.root_dir, 
                            self.filenames[index])
                        
        return path
    
    def _get_audio_sample_label(self, index):
        # return self.annotations.iloc[index, 2]
        # return self.annotations.iloc[index, 0]
        return self.filenames[index]
    
def make_audio_dataloaders(data_dir,classes,sr,silent_reject,amplitude_norm,batch_size,hop_ratio=0.25,tar_l=1.1,l_grain=1024,high_pass_freq=50,num_workers=2):

    print("-------- Creating Dataloaders --------")
    
    # Calculate the hop size based on the ratio
    hop_size = int(hop_ratio*l_grain)
    tar_l = int(tar_l*sr)

    # Cut the target length to that which alligns with grains size
    print("--- Cropping sample lengths from/to:\t",tar_l,tar_l//l_grain*l_grain)
    tar_l = int(tar_l//l_grain*l_grain)

    classes = sorted(classes)
    train_datasets = []
    test_datasets = []
    
    for i, class_label in enumerate(classes):
        files = glob.glob(data_dir+"/*.wav")

        audios = []
        labels = []
        n_rejected = 0
        for file in files:
            data, samplerate = sf.read(file)
            if len(data.shape)>1:
                # convert to mono
                print("--- !!! Convering audio to mono")
                data = data.swapaxes(1, 0)
                data = librosa.to_mono(data)
            if samplerate!=sr:
                print("--- !!! Read samplerate differs from target sample rate, resampling.")
                data = librosa.resample(data, samplerate, sr)

            if len(data)<tar_l:
                # if loaded audio is less than target length, pad with zeros
                data = np.concatenate((data,np.zeros((tar_l-len(data)))))
            else:
                # trim loaded audio to target length
                data = data[:tar_l]
                
            # TODO Lookup what a high pass bi-quad filter is
            # data = torchaudio.functional.highpass_biquad(torch.from_numpy(data),sr,high_pass_freq).numpy()
            
            # Get the power spec
            hann_window = torch.hann_window(l_grain)
            stft = torch.stft(torch.from_numpy(data), n_fft = l_grain,  hop_length=hop_size, window=hann_window, return_complex = True)
            pow_spec = np.abs(stft)**2
        
            # Normalise amplitude between 0 and 1
            pow_spec = (pow_spec - pow_spec.min()) / (pow_spec.max() - pow_spec.min())

            
            audios.append(pow_spec)
            labels.append(i)
        
        audios = torch.from_numpy(np.stack(audios,axis=0)).float()
        labels = torch.from_numpy(np.stack(labels,axis=0)).long()
        print("--- Dataset size:\t\t\t", audios.shape)
        n_grains = audios.shape[2]

        n_samples = len(labels)
        n_train = int(n_samples*0.85)
        dataset = torch.utils.data.TensorDataset(audios,labels)
        train_dataset,test_dataset = torch.utils.data.random_split(dataset, [n_train, n_samples-n_train])
        train_datasets.append(train_dataset)
        test_datasets.append(test_dataset)
    
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    test_dataset = torch.utils.data.ConcatDataset(test_datasets)
    print("--- Dataset train/test sizes:\t\t",len(train_dataset),len(test_dataset))
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_workers)

    print("-------- Done Creating Dataloaders --------")
    
    return train_dataloader,test_dataloader,dataset,tar_l,n_grains,l_grain,hop_size,classes

def make_audio_dataloaders_mel_spec(data_dir,classes,sr,silent_reject,amplitude_norm,batch_size,hop_ratio=0.25,tar_l=1.1,l_grain=2048,high_pass_freq=50,n_mels=128,num_workers=2):

    print("-------- Creating Dataloaders --------")
    
    # Calculate the hop size based on the ratio
    hop_size = int(hop_ratio*l_grain)
    tar_l = int(tar_l*sr)

    # Cut the target length to that which alligns with grains size
    print("--- Cropping sample lengths from/to:\t",tar_l,tar_l//l_grain*l_grain)
    tar_l = int(tar_l//l_grain*l_grain)

    classes = sorted(classes)
    train_datasets = []
    test_datasets = []
    
    for i, class_label in enumerate(classes):
        files = glob.glob(data_dir+"/*.wav")

        audios = []
        labels = []
        n_rejected = 0
        for file in files:
            data, samplerate = sf.read(file)
            if len(data.shape)>1:
                # convert to mono
                print("--- !!! Convering audio to mono")
                data = data.swapaxes(1, 0)
                data = librosa.to_mono(data)
            if samplerate!=sr:
                print("--- !!! Read samplerate differs from target sample rate, resampling.")
                data = librosa.resample(data, samplerate, sr)

            # Mean normalise the grains
            # TODO Removed, but should this be added back?
            # data -= np.mean(data)

            if len(data)<tar_l:
                # if loaded audio is less than target length, pad with zeros
                data = np.concatenate((data,np.zeros((tar_l-len(data)))))
            else:
                # trim loaded audio to target length
                data = data[:tar_l]
                
            # TODO Lookup what a high pass bi-quad filter is
            # NOTE this seems to make a fairly big difference to the reconstructed mel audio
            # data = torchaudio.functional.highpass_biquad(torch.from_numpy(data),sr,high_pass_freq).numpy()
                
            # Normalise amplitude between 0 and 1
            # if amplitude_norm or np.max(np.abs(data))>=1:
            #     data /= np.max(np.abs(data))
            #     data *= 0.9

            print("Data Min: ", data.min())
            print("Data Max: ", data.max())

            # Get the power spec
            mel_spec = librosa.feature.melspectrogram(data, sr = 44100, n_fft = l_grain, hop_length=hop_size, n_mels=n_mels)
        
            # Normalise amplitude between 0 and 1
            # pow_spec = (pow_spec - pow_spec.min()) / (pow_spec.max() - pow_spec.min())

            
            audios.append(mel_spec)
            labels.append(i)
        
        audios = torch.from_numpy(np.stack(audios,axis=0)).float()
        labels = torch.from_numpy(np.stack(labels,axis=0)).long()
        print("--- Dataset size:\t\t\t", audios.shape)
        n_grains = audios.shape[2]

        n_samples = len(labels)
        n_train = int(n_samples*0.85)
        dataset = torch.utils.data.TensorDataset(audios,labels)
        train_dataset,test_dataset = torch.utils.data.random_split(dataset, [n_train, n_samples-n_train])
        train_datasets.append(train_dataset)
        test_datasets.append(test_dataset)
    
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    test_dataset = torch.utils.data.ConcatDataset(test_datasets)
    print("--- Dataset train/test sizes:\t\t",len(train_dataset),len(test_dataset))
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_workers)

    print("-------- Done Creating Dataloaders --------")
    
    return train_dataloader,test_dataloader,dataset,tar_l,n_grains,l_grain,hop_size,classes
