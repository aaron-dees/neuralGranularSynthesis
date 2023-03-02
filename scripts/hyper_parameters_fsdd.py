import torch

# Device to use
# DEVICE = torch.device('mps:0')
DEVICE = torch.device('cpu')

# Mode and directories
TRAIN = False
SAVE_MODEL = False
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
FRAME_SIZE = 512
HOP_LENGTH = 256
DURATION = 0.74 # SECONDS
SAMPLE_RATE = 22050
MONO = True
SPECTROGRAMS_SAVE_DIR = "/Users/adees/Code/neural_granular_synthesis/datasets/fsdd/log_spectograms"
MIN_MAX_VALUES_SAVE_DIR = "/Users/adees/Code/neural_granular_synthesis/datasets/fsdd"
FILES_DIR = "/Users/adees/Code/neural_granular_synthesis/datasets/fsdd/recordings"