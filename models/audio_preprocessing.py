import os
import pickle

import librosa
import numpy as np

class Loader:

     """ Responsible for loadin and audio file """

     def __init__(self, sample_rate, duration, mono):
        self.sample_rate = sample_rate
        self.duration = duration
        self.mono = mono
    
     def load(self, file_path):

        signal, sample_rate = librosa.load(file_path, 
                                sr = self.sample_rate, 
                                duration = self.duration, 
                                mono = self.mono)

        return signal


class Padder:
    """Padder is responsible to apply padding to an array"""

    def __init__(self, mode = "constant"):
        self.mode = mode

    def left_pad(self, array, num_missing_items):
        
        padded_array = np.pad(array,
                                (num_missing_items, 0),
                                mode = self.mode)
        return padded_array 

    def right_pad(self, array, num_missing_items):

        padded_array = np.pad(array,
                                (0, num_missing_items),
                                mode = self.mode)
        return padded_array 


class LogSpectrogramExtractor:
    """LogSpectrogrmExtractor extracts log spectograms in dB from a time series"""

    def __init__(self, frame_size, hop_length):
        self.frame_size = frame_size
        self.hop_length = hop_length
    
    def extract(self, signal):

        stft = librosa.stft(signal,  
                            n_fft = self.frame_size,
                            hop_length = self.hop_length)[:-1]
        # Above returns array fo shape, (1 + frame_size / 2, num_frames)
        # This means if we use 1024 frame size for instance, we will get 513 in first dimension.
        # NOTE: Since we would rather deal with even numbers, we will drop one of the freq bins, 
        # should check what effect this has.
        spectrogram = np.abs(stft)
        log_spectrogram = librosa.amplitude_to_db(spectrogram)
        return log_spectrogram
        



class MinMaxNormaliser:
    """Normalises the array using min max normalisation"""

    def __init__(self, min, max):
        self.min = min
        self.max = max
    
    def normalise(self, array):

        norm_array = (array - array.min()) / (array.max() - array.min())
        norm_array = norm_array * (self.max - self.min) + self.min

        return norm_array

    def denormalise(self, norm_array, original_min, original_max):
        
        array = (norm_array - self.min) / (self.max - self.min)
        array = array * (original_max - original_min) + original_min

        return array

class Saver:

    """Saves feature and min max values"""

    def __init__(self, feature_save_dir, min_max_values_save_dir):
        self.feature_save_dir = feature_save_dir
        self.min_max_values_save_dir = min_max_values_save_dir

    def save_feature(self, feature, file_path):
        save_path = self._generate_save_path(file_path)
        np.save(save_path, feature)

    def save_min_max_values(self, min_max_values):

        save_path = os.path.join(self.min_max_values_save_dir,
                                "min_max_values.pkl")

        self._save(min_max_values, save_path)

    @staticmethod
    def _save(data, save_path):

        with open(save_path, "wb") as f:
            pickle.dump(data, f)

    def _generate_save_path(self, file_path):

        file_name = os.path.split(file_path)[1]
        save_path = os.path.join(self.feature_save_dir, file_name + ".npy")

        return save_path




class PreProcessingPipeline:

    """
    Processes audiofiles in a directory, applying all the following steps to each file:

        1 - load a file
        2 - pad the signal if neccisary
        3 - extracting log spectogram from signal using librosa
        4 - normalise spectorgrams
        5 - save normalised spectogram

    Storing the min max values of all the original log spectograms
    """

    def __init__(self):

        
        self.padder = None
        self.extractor = None
        self.normaliser = None
        self.saver = None
        # Dict of dicts for storing the save path and original min max values of spectorgram
        self.min_max_values = {}
        self._loader = None
        self._num_expected_samples = None

    @property
    def loader(self):
        return self._loader
    
    @loader.setter
    def loader(self, loader):
        self._loader = loader
        self._num_expected_samples = int(loader.sample_rate * loader.duration)
    
    def process(self, audio_files_dir):

        for root, _, files in os.walk(audio_files_dir):
            for file in files:
                file_path = os.path.join(root, file)
                self._process_file(file_path)
                print(f"Processed file {file_path}")
        
        self.saver.save_min_max_values(self.min_max_values)

    def _process_file(self, file_path):

        signal = self.loader.load(file_path)
        if self._is_padding_neccessary(signal):
            signal = self._apply_padding(signal)
        feature = self.extractor.extract(signal)
        norm_feature = self.normaliser.normalise(feature)
        save_path = self.saver.save_feature(norm_feature, file_path)
        self._store_min_max_values(save_path, feature.min(), feature.max())

    def _is_padding_neccessary(self, signal):

        if len(signal) < self._num_expected_samples:
            return True
        return False

    def _apply_padding(self, signal):

        num_missing_samples = self._num_expected_samples - len(signal)
        padded_signal = self.padder.right_pad(signal, num_missing_samples)

        return padded_signal
    
    def _store_min_max_values(self, save_path, min_val, max_val):

        self.min_max_values[save_path] = {
            "min": min_val,
            "max": max_val
        }

if __name__ == "__main__":
    FRAME_SIZE = 512
    HOP_LENGTH = 256
    DURATION = 0.74 # SECONDS
    SAMPLE_RATE = 22050
    MONO = True

    SPECTROGRAMS_SAVE_DIR = "/Users/adees/Code/neural_granular_synthesis/datasets/fsdd/log_spectograms"
    MIN_MAX_VALUES_SAVE_DIR = "/Users/adees/Code/neural_granular_synthesis/datasets/fsdd"
    FILES_DIR = "/Users/adees/Code/neural_granular_synthesis/datasets/fsdd/recordings"

    # instantiate all objects
    loader = Loader(SAMPLE_RATE, DURATION, MONO)
    padder = Padder()
    log_spectrogram_extractor = LogSpectrogramExtractor(FRAME_SIZE, HOP_LENGTH)
    min_max_normaliser = MinMaxNormaliser(0, 1)
    saver = Saver(SPECTROGRAMS_SAVE_DIR, MIN_MAX_VALUES_SAVE_DIR)

    preprocessing_pipeline = PreProcessingPipeline()
    preprocessing_pipeline.loader = loader
    preprocessing_pipeline.padder = padder
    preprocessing_pipeline.extractor = log_spectrogram_extractor
    preprocessing_pipeline.normaliser = min_max_normaliser
    preprocessing_pipeline.saver = saver

    preprocessing_pipeline.process(FILES_DIR)
