"""
1 - load a file
2 - pad the signal if nec
3 - extracting log spectogram from signal using librosa
4 - normalise spectorgrams
5 - save normalised spectogram

PreprocessingPipeline
"""

# import hub
# ds = hub.load("hub://activeloop/spoken_mnist")
import librosa

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
    pass

class LogSpectrogramExtractor:
    pass

class MinMaxNormaliser:
    pass

class Saver:
    pass

class PreProcessingPipeline:
    pass