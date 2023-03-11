import torch

# if torch.backends.mps.is_available():
#     DEVICE = torch.device("mps:0")
# else:
DEVICE = torch.device("cpu") 

# having issues with padding on Apple GPU
DATALOADER_DEVICE = torch.device('cpu')

# Hyper Parameters
BATCH_SIZE = 64
TEST_SIZE = 10
EPOCHS = 20
LEARNING_RATE = 0.0005
LATENT_SIZE = 128
ENV_DIST = 0
BETA = 0.0000001
KERNEL_SIZE = 3

# Audio Processing Parameters
ANNOTATIONS_FILE = "/Users/adees/Code/neural_granular_synthesis/datasets/UrbanSound8K/metadata/UrbanSound8K.csv"
AUDIO_DIR = "/Users/adees/Code/neural_granular_synthesis/datasets/UrbanSound8K/audio"

FRAME_SIZE = 1024
HOP_LENGTH = 512
NUM_MELS = 64
NUM_SAMPLES = 2048 
SAMPLE_RATE = 22050
DURATION = NUM_SAMPLES / SAMPLE_RATE # SECONDS
MONO = True

# Mode and directories
WANDB = False
TRAIN = False
SAVE_CHECKPOINT = False
CHECKPOINT_REGULAIRTY = 5
LOAD_CHECKPOINT = True
VIEW_LATENT = False
SAVE_RECONSTRUCTIONS = True
RECONSTRUCTION_SAVE_DIR = "/Users/adees/Code/neural_granular_synthesis/datasets/fsdd/reconstructions"
# CHECKPOINT_FILE_PATH = f"/Users/adees/Code/neural_granular_synthesis/models/saved_models/checkpoints/waveform_vae_cpu_{EPOCHS}epochs_{BATCH_SIZE}batch_{BETA}beta_{ENV_DIST}envdist_latest.pt"
CHECKPOINT_FILE_PATH = f"/Users/adees/Code/neural_granular_synthesis/models/saved_models/checkpoints/waveform_vae_cpu_{EPOCHS}epochs_{64}batch_{BETA}beta_{ENV_DIST}envdist_latest.pt"
