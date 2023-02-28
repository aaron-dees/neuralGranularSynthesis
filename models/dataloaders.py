import torch
import glob
import numpy as np

class FSDDDataset(torch.utils.data.Dataset):

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