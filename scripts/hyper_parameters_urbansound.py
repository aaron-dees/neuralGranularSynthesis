import torch

# Device to use
DEVICE = torch.device('mps:0')
# DEVICE = torch.device('cpu')
# DEVICE = torch.device('cuda')

# Mode and directories
TRAIN = False
SAVE_MODEL = True
LOAD_MODEL = True
MODEL_PATH = '/Users/adees/Code/neural_granular_synthesis/models/saved_models/fsdd_vae_gpu_100epochs_10batch.pt'
MIN_MAX_VALUES_PATH = "/Users/adees/Code/neural_granular_synthesis/datasets/fsdd/min_max_values.pkl"
RECONSTRUCTION_SAVE_DIR = "/Users/adees/Code/neural_granular_synthesis/datasets/fsdd/reconstructions"

# Hyper Parameters
BATCH_SIZE = 10
EPOCHS = 100
LEARNING_RATE = 0.0005
LATENT_SIZE = 128

# Audio Processing Parameters
ANNOTATIONS_FILE = "/Users/adees/Code/neural_granular_synthesis/datasets/UrbanSound8K/metadata/UrbanSound8K.csv"
AUDIO_DIR = "/Users/adees/Code/neural_granular_synthesis/datasets/UrbanSound8K/audio"

FRAME_SIZE = 1024
HOP_LENGTH = 512
NUM_MELS = 64
NUM_SAMPLES = 22050
SAMPLE_RATE = 22050
DURATION = NUM_SAMPLES / SAMPLE_RATE # SECONDS
MONO = True
SPECTROGRAMS_SAVE_DIR = "/Users/adees/Code/neural_granular_synthesis/datasets/fsdd/log_spectograms"
MIN_MAX_VALUES_SAVE_DIR = "/Users/adees/Code/neural_granular_synthesis/datasets/fsdd"
FILES_DIR = "/Users/adees/Code/neural_granular_synthesis/datasets/fsdd/recordings"