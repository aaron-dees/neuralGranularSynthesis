import torch

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps:0")
else:
    DEVICE = torch.device("cpu") 

# having issues with padding on Apple GPU
DATALOADER_DEVICE = torch.device('cpu')

# Mode and directories
WANDB = False
TRAIN = True
SAVE_MODEL = True
LOAD_MODEL = True
MODEL_PATH = '/Users/adees/Code/neural_granular_synthesis/models/saved_models/usd_vae_gpu_10epochs_37batch.pt'
MIN_MAX_VALUES_PATH = "/Users/adees/Code/neural_granular_synthesis/datasets/fsdd/min_max_values.pkl"
RECONSTRUCTION_SAVE_DIR = "/Users/adees/Code/neural_granular_synthesis/datasets/fsdd/reconstructions"

# Hyper Parameters
BATCH_SIZE = 2
EPOCHS = 20
LEARNING_RATE = 0.0005
LATENT_SIZE = 128
RECONSTRUCTION_LOSS_WEIGHT = 1000000

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