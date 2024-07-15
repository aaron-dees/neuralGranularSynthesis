
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
    
def make_audio_dataloaders(data_dir,classes,sr,silent_reject,amplitude_norm,batch_size,hop_ratio=0.25,tar_l=1.1,l_grain=2048,high_pass_freq=50,num_workers=2, center_pad=False):

    print("-------- Creating Dataloaders --------")
    
    # Calculate the hop size based on the ratio
    hop_size = int(hop_ratio*l_grain)
    tar_l = int(tar_l*sr)

    # Cut the target length to that which alligns with grains size
    print("--- Cropping sample lengths from/to:\t",tar_l,tar_l//l_grain*l_grain)
    tar_l = int(tar_l//l_grain*l_grain)

    # n_grains are the overlapping grains
    #  - This looks like it is based on the hop_ratio
    #  - I understand the multiplication by 4, but not so much the removal of 3
    #  TODO - understand this fromulae 
    print("--- Number of non-overlapping grains:\t",tar_l//l_grain)
    n_grains = 0
    if (hop_ratio*100 == 25) :
        if(center_pad):
            n_grains = 4*((tar_l+l_grain)//l_grain)-3
        else:
            n_grains = 4*(tar_l//l_grain)-3
    elif (hop_ratio*100 == 50) :
        if(center_pad):
            n_grains = 2*(tar_l+l_grain//l_grain)-1
        else:
            n_grains = 2*(tar_l//l_grain)-1
    else :
        print("--- !!! HOP RATIO ENTERED NOT VALID.")
 

    print("--- Number of overlapping grains:\t", n_grains)
    
    classes = sorted(classes)
    train_datasets = []
    test_datasets = []
    
    for i, class_label in enumerate(classes):
        files = glob.glob(data_dir+"/*.wav")

        audios = []
        labels = []
        n_rejected = 0
        for file in files:
            reject = 0
            data, samplerate = sf.read(file)
            print("DATA: ", data.shape)
            if len(data.shape)>1:
                # convert to mono
                print("--- !!! Convering audio to mono")
                data = data.swapaxes(1, 0)
                data = librosa.to_mono(data)
            if samplerate!=sr:
                print("--- !!! Read samplerate differs from target sample rate, resampling.")
                print("Actual sample rate: ", samplerate)
                data = librosa.resample(data, orig_sr=samplerate, target_sr=sr)

            # Mean normalise the grains
            data -= np.mean(data)
            if silent_reject[0]!=0 and np.max(np.abs(data))<silent_reject[0]:
                reject = 1 # peak amplitude is too low
            trim_pos = librosa.effects.trim(data, top_db=60, frame_length=1024, hop_length=128)[1]
            if silent_reject[1]!=0 and (trim_pos[1]-trim_pos[0])<silent_reject[1]*tar_l:
                reject = 1 # non-silent length is too low

            if reject==0:
                if len(data)<tar_l:
                    # if loaded audio is less than target length, pad with zeros
                    data = np.concatenate((data,np.zeros((tar_l-len(data)))))
                else:
                    # trim loaded audio to target length
                    data = data[:tar_l]

                # Pad
                if(center_pad):
                    data = np.pad(data, l_grain//2, 'constant')

                # TODO Lookup what a high pass bi-quad filter is
                # data = torchaudio.functional.highpass_biquad(torch.from_numpy(data),sr,high_pass_freq).numpy()
                
                # Normalise amplitude between 0 and 1
                if amplitude_norm or np.max(np.abs(data))>=1:
                    data /= np.max(np.abs(data))
                    data *= 0.9

                audios.append(data)
                labels.append(i)
            else:
                n_rejected += 1
        
        print("--- Number of audio samples rejected:\t", n_rejected)


        audios = torch.from_numpy(np.stack(audios,axis=0)).float()
        labels = torch.from_numpy(np.stack(labels,axis=0)).long()
        print("--- Dataset size:\t\t\t", audios.shape)

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

    if(center_pad):
        # Add padding onto target length?
        tar_l += l_grain
    
    return train_dataloader,test_dataloader,dataset,tar_l,n_grains,l_grain,hop_size,classes

def make_audio_dataloaders_noPadding(data_dir,classes,sr,silent_reject,amplitude_norm,batch_size,hop_ratio=0.25,tar_l=1.1,l_grain=2048,high_pass_freq=50,num_workers=2):

    print("-------- Creating Dataloaders --------")
    
    # Calculate the hop size based on the ratio
    hop_size = int(hop_ratio*l_grain)
    tar_l = int(tar_l*sr)

    # Cut the target length to that which alligns with grains size
    print("--- Cropping sample lengths from/to:\t",tar_l,tar_l//l_grain*l_grain)
    tar_l = int(tar_l//l_grain*l_grain)

    # n_grains are the overlapping grains
    #  - This looks like it is based on the hop_ratio
    #  - I understand the multiplication by 4, but not so much the removal of 3
    #  TODO - understand this fromulae 
    print("--- Number of non-overlapping grains:\t",tar_l//l_grain)
    n_grains = 0
    if (hop_ratio*100 == 25) :
        n_grains = 4*(tar_l//l_grain)-3
    elif (hop_ratio*100 == 50) :
        n_grains = 2*(tar_l//l_grain)-1
    else :
        print("--- !!! HOP RATIO ENTERED NOT VALID.")
 

    print("--- Number of overlapping grains:\t", n_grains)
    
    classes = sorted(classes)
    train_datasets = []
    test_datasets = []
    
    for i, class_label in enumerate(classes):
        files = glob.glob(data_dir+"/*.wav")

        audios = []
        labels = []
        n_rejected = 0
        for file in files:
            reject = 0
            data, samplerate = sf.read(file)
            print("DATA: ", data.shape)
            if len(data.shape)>1:
                # convert to mono
                print("--- !!! Convering audio to mono")
                data = data.swapaxes(1, 0)
                data = librosa.to_mono(data)
            if samplerate!=sr:
                print("--- !!! Read samplerate differs from target sample rate, resampling.")
                data = librosa.resample(data, orig_sr=samplerate, target_sr=sr)

            # Mean normalise the grains
            data -= np.mean(data)
            if silent_reject[0]!=0 and np.max(np.abs(data))<silent_reject[0]:
                reject = 1 # peak amplitude is too low
            trim_pos = librosa.effects.trim(data, top_db=60, frame_length=1024, hop_length=128)[1]
            if silent_reject[1]!=0 and (trim_pos[1]-trim_pos[0])<silent_reject[1]*tar_l:
                reject = 1 # non-silent length is too low

            if reject==0:
                if len(data)<tar_l:
                    # if loaded audio is less than target length, pad with zeros
                    data = np.concatenate((data,np.zeros((tar_l-len(data)))))
                else:
                    # trim loaded audio to target length
                    data = data[:tar_l]

                # TODO Lookup what a high pass bi-quad filter is
                # data = torchaudio.functional.highpass_biquad(torch.from_numpy(data),sr,high_pass_freq).numpy()
                
                # Normalise amplitude between 0 and 1
                if amplitude_norm or np.max(np.abs(data))>=1:
                    print()
                    data /= np.max(np.abs(data))
                    data *= 0.9

                audios.append(data)
                labels.append(i)
            else:
                n_rejected += 1
        
        print("--- Number of audio samples rejected:\t", n_rejected)


        audios = torch.from_numpy(np.stack(audios,axis=0)).float()
        labels = torch.from_numpy(np.stack(labels,axis=0)).long()
        print("--- Dataset size:\t\t\t", audios.shape)

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
